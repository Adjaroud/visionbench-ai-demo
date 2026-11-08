"""Valide l'image CPU et son interface HTTP, sans benchmark GPU ni appel LLM."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
from urllib.request import ProxyHandler, build_opener
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
LIMITS = ["--read-only", "--tmpfs", "/tmp:rw,nosuid,size=128m", "--memory", "512m", "--cpus", "2",
          "--cap-drop", "ALL", "--security-opt", "no-new-privileges"]


def docker(*args, timeout=90):
    result = subprocess.run(["docker", *args], capture_output=True, encoding="utf-8", timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="visionbench-ai:j6-demo")
    parser.add_argument("--port", type=int, default=8502)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--keep-running", action="store_true")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Choisir un port local entre 1024 et 65535.")
    if args.out.exists():
        parser.error("Le rapport existe deja ; utiliser un nouveau chemin.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    name = "visionbench-demo-" + uuid4().hex[:10]
    started = False
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "image": args.image,
              "success": False, "limits": {"memory_mib": 512, "cpus": 2, "read_only": True,
                                           "tests_network": "none", "live_llm": False}}
    try:
        metadata = json.loads(docker("image", "inspect", args.image))[0]
        report["artifact"] = {"id": metadata["Id"], "size_bytes_reported_by_docker": metadata["Size"],
                              "os": metadata["Os"], "architecture": metadata["Architecture"], "user": metadata["Config"]["User"]}
        report["server_version"] = json.loads(docker("version", "--format", "{{json .Server}}"))["Version"]
        audit = """
import importlib.util, json, os, platform
from pathlib import Path
from visionbench.ui.service import catalogue_path, load_comparison, load_manifest
manifest=load_manifest(Path('/app'))
comparison=load_comparison(Path('/app'))
catalogue_path(Path('/app'), 'resources_100')
assert os.getuid()==10001
assert importlib.util.find_spec('torch') is None
assert importlib.util.find_spec('langgraph') is None
assert not any(Path('/app').rglob('*.pth'))
print(json.dumps({'uid':os.getuid(),'python':platform.python_version(),
                  'verified_data_files':len(set(manifest['files_sha256']) | set(comparison['files_sha256'])),
                  'torch_installed':False,'langgraph_installed':False}))
"""
        report["runtime"] = json.loads(docker("run", "--rm", "--network", "none", *LIMITS,
                                              args.image, "python", "-B", "-c", audit))
        checks = subprocess.run(["docker", "run", "--rm", "--network", "none", *LIMITS,
                                 "--mount", f"type=bind,source={ROOT / 'tests'},target=/app/tests,readonly",
                                 args.image, "python", "-X", "utf8", "-B", "-m", "unittest", "discover",
                                 "-s", "tests", "-p", "test_demo.py", "-v"],
                                capture_output=True, encoding="utf-8", timeout=90)
        report["tests"] = {"exit_code": checks.returncode, "output": checks.stdout + checks.stderr}
        if checks.returncode:
            raise RuntimeError("Les tests d'interface du conteneur echouent. Consulter le rapport.")
        docker("run", "--detach", "--rm", "--name", name, *LIMITS,
               "--publish", f"127.0.0.1:{args.port}:8501", args.image)
        started = True
        opener = build_opener(ProxyHandler({}))
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            try:
                with opener.open(f"http://127.0.0.1:{args.port}/_stcore/health", timeout=2) as response:
                    if response.status != 200 or response.read() != b"ok":
                        raise ValueError("Reponse HTTP inattendue.")
                state = json.loads(docker("inspect", name))[0]["State"]
                if state["Health"]["Status"] == "healthy":
                    break
            except (OSError, ValueError):
                pass
            time.sleep(1)
        else:
            raise RuntimeError("La verification HTTP et la sante Docker n'ont pas reussi dans le delai.")
        report["http"] = {"url": f"http://127.0.0.1:{args.port}", "status": 200, "health_body": "ok"}
        report["docker_health"] = state["Health"]["Status"]
        report["resource_sample"] = json.loads(docker("stats", "--no-stream", "--format", "{{json .}}", name))
        report["container_name"] = name
        report["success"] = True
        print(f"Image validee, tests d'interface reussis et HTTP 200 sur {report['http']['url']}")
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        report["kept_running"] = started and report["success"] and args.keep_running
        if started and not report["kept_running"]:
            docker("stop", name, timeout=30)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
