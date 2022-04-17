# do check
# common.config.constant
#   - S3_BUCKET: point to numerai
# Dockerfile
#   - update path_to_folder

# adapt to your model
# python model_name/predict.py
# can be also model_name/tits.py
# as long as it computes and submits predictions!
python sternburg/predict.py
