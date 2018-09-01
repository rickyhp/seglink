#python visualize_detection_result.py \
#	--image=~/dataset/ICDAR2015/Challenge2.Task123/Challenge2_Test_Task12_Images/ \
#	--gt=~/dataset/ICDAR2015/Challenge2.Task123/Challenge2_Test_Task1_GT/ \
#	--det=~/temp/no-use/seglink_debug_icdar2013/eval/icdar2013_test/model.ckpt-48176/txt/ \
#	--output=~/temp/no-use/seglink_result

#python visualize_detection_result.py \
#	--image=~/dataset/ICDAR2015/Challenge2.Task123/Challenge2_Training_Task12_Images/ \
#	--gt=~/dataset/ICDAR2015/Challenge2.Task123/Challenge2_Training_Task1_GT/ \
#	--det=~/temp/no-use/seglink_debug_icdar2013/eval/icdar2013_train/model.ckpt-48176/txt/ \
#	--output=~/temp/no-use/seglink_result

	
python visualize_detection_result.py \
	--image=~/code/icdar2015/ch4_test_images/ \
	--gt=~/code/icdar2015/ch4_test_localization_transcription_gt/ \
	--det=~/code/seglink/seglink-512/model.ckpt-217867/test/icdar2015_test/model.ckpt-217867/seg_link_conf_th_0.800000_0.500000/txt \
	--output=~/code/seglink/output/seglink_result_512_train
	