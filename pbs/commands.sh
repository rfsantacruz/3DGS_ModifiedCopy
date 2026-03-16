qsub -N GSVanilla -v DATASET=001_PKA_3393_KS0,GS_EXTRA_ARGS="" gs_kfold_mmor.aqua ; qsub -N GSVanilla -v DATASET=001_PKA_3393_KS01,GS_EXTRA_ARGS="" gs_kfold_mmor.aqua ; qsub -N GSVanilla -v DATASET=001_PKA_3393_KS03,GS_EXTRA_ARGS="" gs_kfold_mmor.aqua ; qsub -N GSVanilla -v DATASET=001_PKA_3393_KS0_G2k_VD001,GS_EXTRA_ARGS="" gs_kfold_mmor.aqua ; qsub -N GSVanilla -v DATASET=001_PKA_3393_KS0_G2k_VD005,GS_EXTRA_ARGS="" gs_kfold_mmor.aqua ; qsub -N GSVanilla -v DATASET=001_PKA_3393_KS0_G2k_VD01,GS_EXTRA_ARGS="" gs_kfold_mmor.aqua ; qsub -N GSSensorDepth -v DATASET=001_PKA_3393_KS0,GS_EXTRA_ARGS="--sensor_depths sensor_depth" gs_kfold_mmor.aqua ; qsub -N GSSensorDepth -v DATASET=001_PKA_3393_KS01,GS_EXTRA_ARGS="--sensor_depths sensor_depth" gs_kfold_mmor.aqua ; qsub -N GSSensorDepth -v DATASET=001_PKA_3393_KS03,GS_EXTRA_ARGS="--sensor_depths sensor_depth" gs_kfold_mmor.aqua ; qsub -N GSSensorDepth -v DATASET=001_PKA_3393_KS0_G2k_VD001,GS_EXTRA_ARGS="--sensor_depths sensor_depth" gs_kfold_mmor.aqua ; qsub -N GSSensorDepth -v DATASET=001_PKA_3393_KS0_G2k_VD005,GS_EXTRA_ARGS="--sensor_depths sensor_depth" gs_kfold_mmor.aqua ; qsub -N GSSensorDepth -v DATASET=001_PKA_3393_KS0_G2k_VD01,GS_EXTRA_ARGS="--sensor_depths sensor_depth" gs_kfold_mmor.aqua ; qsub -N GSMDEDepth -v DATASET=001_PKA_3393_KS0,GS_EXTRA_ARGS="--sensor_depths marigold_depth" gs_kfold_mmor.aqua ; qsub -N GSMDEDepth -v DATASET=001_PKA_3393_KS01,GS_EXTRA_ARGS="--sensor_depths marigold_depth" gs_kfold_mmor.aqua ; qsub -N GSMDEDepth -v DATASET=001_PKA_3393_KS03,GS_EXTRA_ARGS="--sensor_depths marigold_depth" gs_kfold_mmor.aqua ; qsub -N GSMDEDepth -v DATASET=001_PKA_3393_KS0_G2k_VD001,GS_EXTRA_ARGS="--sensor_depths marigold_depth" gs_kfold_mmor.aqua ; qsub -N GSMDEDepth -v DATASET=001_PKA_3393_KS0_G2k_VD005,GS_EXTRA_ARGS="--sensor_depths marigold_depth" gs_kfold_mmor.aqua ; qsub -N GSMDEDepth -v DATASET=001_PKA_3393_KS0_G2k_VD01,GS_EXTRA_ARGS="--sensor_depths marigold_depth" gs_kfold_mmor.aqua ; 

qsub -N GSSensorDepthMDEGrad -v DATASET=001_PKA_3393_KS0,GS_EXTRA_ARGS="--sensor_depths sensor_depth --ml_depths marigold_depth" gs_kfold_mmor.aqua ; qsub -N GSSensorDepthMDEGrad -v DATASET=001_PKA_3393_KS01,GS_EXTRA_ARGS="--sensor_depths sensor_depth --ml_depths marigold_depth" gs_kfold_mmor.aqua ; qsub -N GSSensorDepthMDEGrad -v DATASET=001_PKA_3393_KS03,GS_EXTRA_ARGS="--sensor_depths sensor_depth --ml_depths marigold_depth" gs_kfold_mmor.aqua ; qsub -N GSSensorDepthMDEGrad -v DATASET=001_PKA_3393_KS0_G2k_VD001,GS_EXTRA_ARGS="--sensor_depths sensor_depth --ml_depths marigold_depth" gs_kfold_mmor.aqua ; qsub -N GSSensorDepthMDEGrad -v DATASET=001_PKA_3393_KS0_G2k_VD005,GS_EXTRA_ARGS="--sensor_depths sensor_depth --ml_depths marigold_depth" gs_kfold_mmor.aqua ; qsub -N GSSensorDepthMDEGrad -v DATASET=001_PKA_3393_KS0_G2k_VD01,GS_EXTRA_ARGS="--sensor_depths sensor_depth --ml_depths marigold_depth" gs_kfold_mmor.aqua ; 


 
qsub -N GSVanillaLarge -v DATASET=001_PKA_3393_LARGE,GS_EXTRA_ARGS="" gs_kfold_mmor.aqua ; 
qsub -N GSVanillaMed -v DATASET=001_PKA_3393_MED,GS_EXTRA_ARGS="" gs_kfold_mmor.aqua ; 
qsub -N GSVanillaSmall -v DATASET=001_PKA_3393_SMALL,GS_EXTRA_ARGS="" gs_kfold_mmor.aqua ; 
qsub -N GSVanillaLargeFixed -v DATASET=001_PKA_3393_LARGE,GS_EXTRA_ARGS="--densify_from_iter 10000" gs_kfold_mmor.aqua ; 
qsub -N GSVanillaMedFixed -v DATASET=001_PKA_3393_MED,GS_EXTRA_ARGS="--densify_from_iter 10000" gs_kfold_mmor.aqua ; 
qsub -N GSVanillaSmallFixed -v DATASET=001_PKA_3393_SMALL,GS_EXTRA_ARGS="--densify_from_iter 10000" gs_kfold_mmor.aqua ; 


# isometric scaling
qsub -N GSVanillaMedIso -v DATASET=001_PKA_3393_MED,GS_EXTRA_ARGS="--isotropic_scaling" gs_kfold_mmor.aqua ; 
qsub -N GSVanillaMedFixedIso -v DATASET=001_PKA_3393_MED,GS_EXTRA_ARGS="--densify_from_iter 10000 --isotropic_scaling" gs_kfold_mmor.aqua ; 


# dropout
qsub -N GSVanillaLargeDropout02 -v DATASET=001_PKA_3393_LARGE,GS_EXTRA_ARGS="--drop_gaussian_rate 0.2" gs_kfold_mmor.aqua ; 
qsub -N GSVanillaLargeFixedDropout02 -v DATASET=001_PKA_3393_LARGE,GS_EXTRA_ARGS="--densify_from_iter 10000 --drop_gaussian_rate 0.2" gs_kfold_mmor.aqua ; 
qsub -N GSVanillaLargeDropout04 -v DATASET=001_PKA_3393_LARGE,GS_EXTRA_ARGS="--drop_gaussian_rate 0.4" gs_kfold_mmor.aqua ; 
qsub -N GSVanillaLargeFixedDropout04 -v DATASET=001_PKA_3393_LARGE,GS_EXTRA_ARGS="--densify_from_iter 10000 --drop_gaussian_rate 0.4" gs_kfold_mmor.aqua ; 

# losses
qsub -N GSVanillaLargeFixedDropout04L1SensorDepth -v DATASET=001_PKA_3393_LARGE,GS_EXTRA_ARGS="--densify_from_iter 10000 --drop_gaussian_rate 0.4 --sensor_depths sensor_depth" gs_kfold_mmor.aqua ; 
qsub -N GSVanillaLargeFixedDropout04L1MDE -v DATASET=001_PKA_3393_LARGE,GS_EXTRA_ARGS="--densify_from_iter 10000 --drop_gaussian_rate 0.4 --sensor_depths marigold_depth" gs_kfold_mmor.aqua ; 
qsub -N GSVanillaLargeFixedDropout04L1SensorDepthGradMDE -v DATASET=001_PKA_3393_LARGE,GS_EXTRA_ARGS="--densify_from_iter 10000 --drop_gaussian_rate 0.4 --sensor_depths sensor_depth --ml_depths marigold_depth" gs_kfold_mmor.aqua ; 
qsub -N GSVanillaLargeFixedDropout04L1MDEGradMDE -v DATASET=001_PKA_3393_LARGE,GS_EXTRA_ARGS="--densify_from_iter 10000 --drop_gaussian_rate 0.4 --sensor_depths marigold_depth --ml_depths marigold_depth" gs_kfold_mmor.aqua ; 


# random initialization
qsub -N GSVanillaLargeRandom -v DATASET=001_PKA_3393_LARGE,GS_EXTRA_ARGS="--init_random_points" gs_kfold_mmor.aqua ; 
qsub -N GSVanillaMedRandom -v DATASET=001_PKA_3393_MED,GS_EXTRA_ARGS="--init_random_points" gs_kfold_mmor.aqua ; 
qsub -N GSVanillaSmallRandom -v DATASET=001_PKA_3393_SMALL,GS_EXTRA_ARGS="--init_random_points" gs_kfold_mmor.aqua ; 

# 4D OR
qsub -N 4DORFixedDropout04L1SensorDepthGradMDE gs_kfold_frames.aqua ; 

# control densifications
qsub -N 4DORDensificationStudy1 -v INITIAL=500,INTERVAL=500,RESET=2500 gs_kfold_frames.aqua ;  
qsub -N 4DORDensificationStudy2 -v INITIAL=500,INTERVAL=1000,RESET=2500 gs_kfold_frames.aqua ;  
qsub -N 4DORDensificationStudy3 -v INITIAL=500,INTERVAL=1500,RESET=2500 gs_kfold_frames.aqua ; 
qsub -N 4DORDensificationStudy4 -v INITIAL=500,INTERVAL=2000,RESET=30000 gs_kfold_frames.aqua ; 
qsub -N 4DORDensificationStudy5 -v INITIAL=500,INTERVAL=2500,RESET=30000 gs_kfold_frames.aqua ; 



# paper overfit figure
qsub -N PaperOverfitGSVanilla -v GS_EXTRA_ARGS="" gs_kfold_frames.aqua ; 
qsub -N PaperOverfitGSOurs -v GS_EXTRA_ARGS="--sensor_depths sensor_depth --ml_depths ml_depth --drop_gaussian_rate 0.3 --densify_from_iter 1500 --densification_interval 500 --opacity_reset_interval 1800 --densify_grad_threshold 0.0039 --percent_dense 0.0026 --depth_l1_weight_init 2.533 --depth_l1_weight_final 0.1546" gs_kfold_frames.aqua ; 

# generate dataset for 3dgs
python 4DOR.py /home/fonseca2/dataset/4D-OR/export_holistic_take1_processed 8 /home/fonseca2/dataset/4D-OR-Prep/take1_frame8/
python 4DOR.py /home/fonseca2/dataset/4D-OR/export_holistic_take1_processed 91 /home/fonseca2/dataset/4D-OR-Prep/take1_frame91/
python 4DOR.py /home/fonseca2/dataset/4D-OR/export_holistic_take1_processed 174 /home/fonseca2/dataset/4D-OR-Prep/take1_frame174/
python 4DOR.py /home/fonseca2/dataset/4D-OR/export_holistic_take1_processed 258 /home/fonseca2/dataset/4D-OR-Prep/take1_frame258/
python 4DOR.py /home/fonseca2/dataset/4D-OR/export_holistic_take1_processed 341 /home/fonseca2/dataset/4D-OR-Prep/take1_frame341/
python 4DOR.py /home/fonseca2/dataset/4D-OR/export_holistic_take1_processed 424 /home/fonseca2/dataset/4D-OR-Prep/take1_frame424/
python 4DOR.py /home/fonseca2/dataset/4D-OR/export_holistic_take1_processed 507 /home/fonseca2/dataset/4D-OR-Prep/take1_frame507/
python 4DOR.py /home/fonseca2/dataset/4D-OR/export_holistic_take1_processed 591 /home/fonseca2/dataset/4D-OR-Prep/take1_frame591/
python 4DOR.py /home/fonseca2/dataset/4D-OR/export_holistic_take1_processed 674 /home/fonseca2/dataset/4D-OR-Prep/take1_frame674/
python 4DOR.py /home/fonseca2/dataset/4D-OR/export_holistic_take1_processed 757 /home/fonseca2/dataset/4D-OR-Prep/take1_frame757/


# all timestamps
# paper comparison figure
qsub -N PaperCompGSVanilla -v GS_EXTRA_ARGS="" gs_kfold_frames.aqua ; 
qsub -N PaperCompGSOurs -v GS_EXTRA_ARGS="--sensor_depths sensor_depth --ml_depths ml_depth --drop_gaussian_rate 0.3 --densify_from_iter 1500 --densification_interval 500 --opacity_reset_interval 1800 --densify_grad_threshold 0.0039 --percent_dense 0.0026 --depth_l1_weight_init 2.533 --depth_l1_weight_final 0.1546" gs_kfold_frames.aqua ; 

# for rendrgin results
qsub -N PaperQualiGSVanilla -v GS_EXTRA_ARGS="" gs_frames.aqua ; 
qsub -N PaperQualiGSOurs -v GS_EXTRA_ARGS="--sensor_depths sensor_depth --ml_depths ml_depth --drop_gaussian_rate 0.3 --densify_from_iter 1500 --densification_interval 500 --opacity_reset_interval 1800 --densify_grad_threshold 0.0039 --percent_dense 0.0026 --depth_l1_weight_init 2.533 --depth_l1_weight_final 0.1546" gs_frames.aqua ; 
