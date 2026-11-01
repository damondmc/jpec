# pipelines

The `jpec` command, installed with the package: one subcommand per step of a search or an
injection run over real SFTs, and `time-search` to measure the rates jobs are priced at. For `check-sft`, `search-setup` and
`injection-setup`, `--output` is the JSON file written, by default under `setup/`; for `search`,
`time-search` and `sensitivity` it is the directory the JSON result is written into, by default
`results/`. Each command also writes a time-stamped log of its run next to its output. The defaults
are relative to the directory a command is run from, so a run lives in its own directory or
repository, not in this one.

| command | reads | produces |
|---|---|---|
| `jpec check-sft` | a glob of SFT files over any number of narrow bands | their detectors, timestamps, SFT length and window, once they are checked to agree band to band |
| `jpec search-setup` | a search config, that metadata, and optionally a `time-search` result | the lattice's steps from the metric, and the f0 intervals the search is cut into, each sized in memory and, with the timing, priced in hours |
| `jpec search` | the setup | the loudest templates of each job |
| `jpec injection-setup` | an injection config and the metadata | the band, the bank at one mismatch, and the thresholds the config gives |
| `jpec sensitivity` | that setup | the sensitivity depth at the setup's mismatch for each threshold, with every injected source and whether it was recovered at each $h_0$ |
| `jpec time-search` | a search setup | seconds per call of each stage of the search loop on one sky point, and the rates `search-setup` prices jobs with |

A search on Gaussian noise runs the same way, on SFT files written first with
`gaussian_noise_series_sft()` and `write_sfts()`: noise on the run's own timestamps, overlapping
SFTs sharing their samples as the recorded ones do.

Every setup and result records the jpec version it was made with under `jpec`; bump the
version in `pyproject.toml` before a production run.

## Configs

[`examples/search.yaml`](../../../examples/search.yaml) and
[`examples/injection.yaml`](../../../examples/injection.yaml) are one search and one injection
run at 204 Hz over the O4a box, $a_p$ 5 to 15 ls and $P$ 7 to 15 d; copy them into the run's
directory and set `sft_glob` to the files.

## Run

From the run's directory, with jpec installed into its environment:

```
jpec check-sft "data/*.sft" --window tukey 0.5 --output setup/sft_metadata.json
jpec search-setup search.yaml --sft-metadata setup/sft_metadata.json --timing results/time_search_gpu_<host>.json
jpec search --setup setup/search_setup.json --platform gpu
```

The hours of a job come from a `time-search` result on the machine the jobs run on. On a new
machine, make the setup without one, which cuts the jobs on memory alone, time its first job,
and make the setup again with that timing:

```
jpec search-setup search.yaml --sft-metadata setup/sft_metadata.json
jpec time-search --setup setup/search_setup.json --platform gpu
jpec search-setup search.yaml --sft-metadata setup/sft_metadata.json --timing results/time_search_gpu_<host>.json
```

`python -m jpec.pipelines.cli` runs the same command as `jpec`.
