import vmcnet.train.runners as runners
import os
import argparse
import logging

logging.basicConfig(level=logging.INFO)

def vmc_statistics() -> None:
    # 从命令行读取参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_energies_file_path", type=str, required=True)
    parser.add_argument("--output_file_path", type=str, required=True)
    parser.add_argument("--nchains", type=int, required=True)
    parser.add_argument("--walkers", type=int, required=True)
    parser.add_argument("--cut", type=int, required=True)
    args = parser.parse_args()

    # 直接用传进来的变量
    local_energies_file_path = args.local_energies_file_path
    output_file_path = args.output_file_path
    nchains = args.nchains
    walkers = args.walkers
    cut = args.cut
    repeat_single_mol = False

    output_dir, output_filename = os.path.split(os.path.abspath(output_file_path))
    runners._compute_and_save_energy_statistics(
        local_energies_file_path, output_dir, output_filename,
        nchains, walkers, repeat_single_mol, cut
    )
    logging.info("Done!")

if __name__ == "__main__":
    vmc_statistics()