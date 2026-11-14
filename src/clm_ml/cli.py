"""
The ``clm-ml`` command. An experiment is a folder with run.yaml, domain.yaml, model.yaml and
params.yaml (decision 91; docs/design/experiment_folders.md).

    clm-ml new DIR --from COMPSET            write a complete experiment folder from a compset
        [--machine derecho --account ACCOUNT [--device gpu --devices N --walltime HH:MM:SS]]
                                            (also submit.pbs, a batch job for the folder)
    clm-ml check DIR [--set ...]            validate the four files and print a summary
    clm-ml run DIR [--set file.key=value]   run the model and write the outputs
    clm-ml upgrade DIR                      rewrite a folder with every current setting
    clm-ml convert NAMELIST -o DIR          write an experiment folder from a Fortran namelist
    clm-ml params SOURCE -o PARAMS.yaml     write a complete params.yaml from a CLM parameter
                                            file (.nc), or copy a params file or compset

``--set`` changes one setting without editing the files, the file named first:
``--set 'run.time.length=7 days'``, ``--set model.processes.stomatal_conductance.scheme=medlyn``,
``--set params.broadleaf_deciduous_temperate_tree.vcmax25=100``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from clm_ml.io.yaml_schema import ConfigError


def main(argv: list[str] | None = None) -> int:
    from clm_ml.experiment import compset_names

    parser = argparse.ArgumentParser(prog="clm-ml", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (("run", "run the model"), ("check", "validate and summarize")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("folder", type=Path, help="experiment folder")
        p.add_argument("--set", dest="overrides", action="append", default=[],
                       metavar="FILE.KEY=VALUE", help="change one setting (repeatable)")  # fmt: skip

    p = sub.add_parser("new", help="write a complete experiment folder from a compset")
    p.add_argument("folder", type=Path)
    p.add_argument("--from", dest="compset", required=True,
                   help=f"compset: {', '.join(compset_names())}")  # fmt: skip
    p.add_argument("--force", action="store_true", help="overwrite the folder's files")
    p.add_argument("--machine", help="derecho: also write submit.pbs, a batch job for the folder")
    p.add_argument("--device", choices=("gpu", "cpu"), default="gpu",
                   help="what the job runs on (default gpu)")  # fmt: skip
    p.add_argument("--devices", type=int, help="number of devices (run.execution.devices)")
    p.add_argument("--account", help="project account the job is charged to (derecho)")
    p.add_argument("--walltime", default="04:00:00", help="job wall-clock limit (default 04:00:00)")

    p = sub.add_parser("upgrade", help="rewrite a folder with every current setting")
    p.add_argument("folder", type=Path)

    p = sub.add_parser("convert", help="convert a Fortran-style namelist to an experiment folder")
    p.add_argument("namelist", type=Path)
    p.add_argument("-o", "--out-dir", type=Path, required=True)

    p = sub.add_parser("params", help="write a complete params.yaml (from a CLM .nc file)")
    p.add_argument("source", help="CLM parameter file (.nc), params file or compset name")
    p.add_argument("-o", "--out", type=Path, required=True)
    p.add_argument("--base", default="clm5_defaults",
                   help="params file for what a CLM file lacks (default clm5_defaults)")  # fmt: skip

    args = parser.parse_args(argv)
    try:
        if args.command == "new":
            from clm_ml.experiment import new_experiment

            folder = new_experiment(
                args.folder, args.compset, force=args.force, machine=args.machine,
                device=args.device, devices=args.devices, account=args.account,
                walltime=args.walltime,
            )  # fmt: skip
            print(f"wrote {folder}")
            return 0
        if args.command == "upgrade":
            from clm_ml.experiment import upgrade_experiment

            for line in upgrade_experiment(args.folder):
                print(line)
            return 0
        if args.command == "params":
            from clm_ml.params_convert import write_params_yaml

            print(f"wrote {write_params_yaml(args.source, args.out, args.base)}")
            return 0
        if args.command == "convert":
            from clm_ml.io.namelist import convert_namelist

            print(f"wrote {convert_namelist(args.namelist, args.out_dir)}")
            return 0

        from clm_ml.experiment import load_experiment

        exp = load_experiment(args.folder, args.overrides)
        print(exp.summary())
        if args.command == "run":
            exp.execute()
            print(f"Outputs written to {exp.output_directory}")
        return 0
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
