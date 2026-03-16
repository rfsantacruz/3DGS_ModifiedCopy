import optuna
from joblib import Parallel, delayed
import subprocess
import json
import os, glob
import numpy as np
import shutil
import time

PATH_CMD = "cd /home/fonseca2/gaussian_splatting/gaussian-splatting/;"
STUDY_PATH = "/home/fonseca2/gaussian_splatting/gaussian-splatting/pbs/opt/optuna_gsplat_pareto_joblib.db"
DATASET_ROOT = "/home/fonseca2/dataset/4D-OR-Prep/take1_frame424/"
OUTPUT_ROOT = "/home/fonseca2/gaussian_splatting/gaussian-splatting/output/optuna/"
OUTPUT_ROOT = os.path.join(OUTPUT_ROOT, "RUN_" + str(int(time.time())))
if not os.path.exists(OUTPUT_ROOT):
    os.makedirs(OUTPUT_ROOT)
ITES = [100, 200, 500, 1000, 1500, 2000]


def run_fold(test_cam_id, params_str):    
    """Run training + rendering + metrics for one fold."""    

    # --- Run training ---
    train_cameras = ' '.join([str(i) for i in range(1, 7) if i != test_cam_id])
    train_cmd_str = f"{PATH_CMD} python train.py -s {DATASET_ROOT} -m {OUTPUT_ROOT}/{test_cam_id} -r 1 --train_cam_ids \"{train_cameras}\" --test_iterations 10000 30000 --save_iterations {' '.join(map(str, ITES))} --disable_viewer --iterations 2000 --sensor_depths sensor_depth --ml_depths render_depth {params_str} ;"
    subprocess.run(train_cmd_str, shell=True, check=True)
    time.sleep(5)  # ensure file system is synced

    # --- Run rendering and metrics ---
    for it in ITES:
        render_cmd_str = f"{PATH_CMD} python render.py -m {OUTPUT_ROOT}/{test_cam_id} --iteration {it} ;"
        subprocess.run(render_cmd_str, shell=True, check=True)
        time.sleep(5)  # ensure file system is synced
    
    # --- Compute metrics ---
    metrics_cmd_str = f"{PATH_CMD} python metrics.py -m {OUTPUT_ROOT}/{test_cam_id} ;"
    subprocess.run(metrics_cmd_str, shell=True, check=True)
    time.sleep(5)  # ensure file system is synced

    
    # --- Gather results ---
    test_psnrs = []
    test_metrics_file = f"{OUTPUT_ROOT}/{test_cam_id}/results_test.json"
    with open(test_metrics_file) as f:
        test_metrics = json.load(f)
        for it in ITES:
            test_psnrs.append(float(test_metrics[f"ours_{it}"]['PSNR']))
    
    train_psnrs = []
    train_metrics_file = f"{OUTPUT_ROOT}/{test_cam_id}/results_train.json"
    with open(train_metrics_file) as f:
        train_metrics = json.load(f)
        for it in ITES:
            train_psnrs.append(float(train_metrics[f"ours_{it}"]['PSNR']))


    # --- Clean up output directory ---
    shutil.rmtree(f"{OUTPUT_ROOT}/{test_cam_id}")
    
    return test_psnrs, train_psnrs


def objective(trial):
    # --- Suggest hyperparameters ---
    
    drop_rate = trial.suggest_float("DROP_RATE", 0.1, 0.9, step=0.05)
    
    dens_from_ite = trial.suggest_int("DENS_INIT", 100, 2000, step=50)
    dens_interval = trial.suggest_int("DENS_INTERVAL", 50, 2000, step=50)
    dens_opacity_reset = trial.suggest_int("DENS_OPACITY_RESET", 250, 2000, step=50)
    dens_grad_threshold = trial.suggest_float("DENS_GRAD_THRESH", 0.0001, 0.01)
    dense_percent = trial.suggest_float("DENSE_PERCENT", 0.001, 0.1)
    
    depth_l1_init = trial.suggest_float("DEPTH_L1_INIT", 0.1, 5.0, log=True)
    depth_l1_final = trial.suggest_float("DEPTH_L1_FINAL", 0.1, 5.0, log=True)


    params_str = f"--drop_gaussian_rate {drop_rate} --densify_from_iter {dens_from_ite} --densification_interval {dens_interval} --opacity_reset_interval {dens_opacity_reset} "
    params_str += f"--densify_grad_threshold {dens_grad_threshold} --percent_dense {dense_percent} --depth_l1_weight_init {depth_l1_init} --depth_l1_weight_final {depth_l1_final}"

    # --- Run folds in parallel ---
    results = Parallel(n_jobs=1)(
        delayed(run_fold)(test_cam_id=test_cam, params_str=params_str)
        for test_cam in range(1,7)
    )
    assert len(results) == 6, "Some folds failed."
    assert [len(train_psnrs) == len(ITES) and len(test_psnrs) == len(ITES) for test_psnrs, train_psnrs in results], "Some folds failed."

    # --- Aggregate results of different folds and compute final metrics ---
    test_psnrs_per_fold_ite, train_psnrs_per_fold_ite = np.vstack([x for x, _ in results]), np.vstack([x for _, x in results])
    assert test_psnrs_per_fold_ite.shape == (6, len(ITES)) and train_psnrs_per_fold_ite.shape == (6, len(ITES)), "Some folds failed."
    test_mean_psnr_ite, train_mean_psnr_ite = np.mean(test_psnrs_per_fold_ite, axis=0), np.mean(train_psnrs_per_fold_ite, axis=0)
    best_ite = np.argmax(test_mean_psnr_ite)
    test_mean_psnr, train_mean_psnr = test_mean_psnr_ite[best_ite], train_mean_psnr_ite[best_ite]

    # result of trial
    trial.set_user_attr("best_iteration", int(best_ite))
    return test_mean_psnr, train_mean_psnr


if __name__ == "__main__":    

    # --- Multi-objective study ---
    study = optuna.load_study(study_name="gsplat_pareto_joblib", storage=f"sqlite:///{STUDY_PATH}")
    study.optimize(objective, n_trials=1000)
