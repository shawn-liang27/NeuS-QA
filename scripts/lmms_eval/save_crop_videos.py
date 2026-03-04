from nsvqa.datamanager.longvideobench import *
from nsvqa.datamanager.video_mme import *
from nsvqa.datamanager.mlvu import *

if __name__ == "__main__":
    nsvs_dir = "/usr/homes/sgl57/NeuS-VLM/NeuS-QA/experiment_results/nsvs_improved/ablation/ablation_adaptive_1"

    benchmark = "lvb"
    data_dir = "/usr/homes/sgl57/.data/LongVideoBench/"
    burned_dir = "/usr/homes/sgl57/.data/LongVideoBench/burn-subtitles/T3E_E3E_T3O_O3O_mix_2026_01_14_21_55"
    postprocess_dir = "/usr/homes/sgl57/NeuS-VLM/NeuS-QA/experiment_results/nsvs_improved/InternVL2-8B/rt-neus/experiment_6/split_4/postprocess_output/postprocess_output_4.json"
    category = "E3E"
    data_loader = LongVideoBench(dataset_path=data_dir, burned_path=burned_dir, categories=category, postprocess_dir=postprocess_dir)

    nsvs_path = "/usr/homes/sgl57/NeuS-VLM/NeuS-QA/experiment_results/nsvs_improved/InternVL2-8B/rt-neus/experiment_6/split_4/nsvqa_output/nsvqa_output_4.json"

    data_loader.postprocess_data(nsvs_path=nsvs_path)