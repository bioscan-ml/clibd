import json

if __name__ == '__main__':
    # Read the json file to dictionary
    path = "/localhome/zmgong/second_ssd/projects/bioscan-clip/extracted_embedding/bioscan_1m/image_dna_text_4gpu_with_pre_trained_image_encoder_trained_with_simclr_style/per_class_acc_test.json"
    with open(path, 'r') as f:
        acc_dict = json.load(f)
