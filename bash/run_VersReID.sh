# Train the ReID Bank
python multi_scene_train.py --config_file configs/ReID-Bank.yml MODEL.DEVICE_ID "('0')" MODEL.PRETRAIN_PATH ckpts/dino_mpda_vit_b_lupb.pth OUTPUT_DIR logs/ReID-Bank/
mv logs/ReID-Bank/train_log.txt logs/ReID-Bank/ReID-Bank_train.txt

# Test the ReID Bank (single test)
python multi_scene_single_test.py --config_file configs/ReID-Bank.yml MODEL.DEVICE_ID "('0')" MODEL.PRETRAIN_CHOICE none TEST.WEIGHT logs/ReID-Bank/transformer_120.pth OUTPUT_DIR logs/ReID-Bank/
mv logs/ReID-Bank/test_log.txt logs/ReID-Bank/ReID-Bank_single_test.txt

# Train the ReID Bank (joint test)
python multi_scene_joint_test.py --config_file configs/ReID-Bank.yml MODEL.DEVICE_ID "('0')" MODEL.PRETRAIN_CHOICE none TEST.WEIGHT logs/ReID-Bank/transformer_120.pth OUTPUT_DIR logs/ReID-Bank/
mv logs/ReID-Bank/test_log.txt logs/ReID-Bank/ReID-Bank_joint_test.txt

# ----------------------------------------------------------

# Train the V-Branch
python multi_scene_distillate.py --config_file configs/V-Branch.yml MODEL.DEVICE_ID "('0')" MODEL.PRETRAIN_PATH logs/ReID-Bank/transformer_120.pth OUTPUT_DIR logs/V-Branch/
mv logs/V-Branch/train_log.txt logs/V-Branch/V-Branch_train.txt

# Test the V-Branch (single test)
python multi_scene_single_test.py --config_file configs/V-Branch.yml MODEL.DEVICE_ID "('0')" MODEL.PRETRAIN_CHOICE none TEST.WEIGHT logs/V-Branch/transformer_120.pth MODEL.AUX_LOSS True OUTPUT_DIR logs/V-Branch/
mv logs/V-Branch/test_log.txt logs/V-Branch/V-Branch_single_test.txt

# Test the V-Branch (joint test)
python multi_scene_joint_test.py --config_file configs/V-Branch.yml MODEL.DEVICE_ID "('0')" MODEL.PRETRAIN_CHOICE none TEST.WEIGHT logs/V-Branch/transformer_120.pth MODEL.AUX_LOSS True OUTPUT_DIR logs/V-Branch/
mv logs/V-Branch/test_log.txt logs/V-Branch/V-Branch_joint_test.txt