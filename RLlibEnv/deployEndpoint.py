import boto3
import sagemaker
import os
import numpy as np
import urllib.request
from time import time

from sagemaker.tensorflow.serving import Model

if __name__ == "__main__":
    endpoint_instance_type = "SAGEMAKER_INFERENCE_INSTANCE_TYPE"
    sage_session = sagemaker.session.Session()
    s3_bucket = sage_session.default_bucket()
    role = sagemaker.get_execution_role()

    s3_client = boto3.client('s3')
    s3_client.upload_file("RLlibEnv/inference/model.tar.gz", s3_bucket, 
                          "battlesnake-aws/pretrainedmodels/model.tar.gz")

    model_data = "s3://{}/battlesnake-aws/pretrainedmodels/model.tar.gz".format(s3_bucket)
    print("Make an endpoint with {}".format(model_data))
    

    model = Model(model_data=model_data,
                  role=role,
                  entry_point="inference.py",
                  source_dir='RLlibEnv/inference/inference_src',
                  framework_version='2.1.0',
                  name="battlesnake-rllib",
                 )

    # Deploy an inference endpoint
    predictor = model.deploy(initial_instance_count=1, instance_type=endpoint_instance_type, 
                             endpoint_name='battlesnake-endpoint')
    
    state = np.zeros(shape=(1, 21, 21, 6), dtype=np.float32).tolist()

    health_dict = {0: 50, 1: 50}
    json = {"turn": 4,
            "board": {
                    "height": 11,
                    "width": 11,
                    "food": [],
                    "snakes": []
                    },
                "you": {
                    "id": "snake-id-string",
                    "name": "Sneky Snek",
                    "health": 90,
                    "body": [{"x": 1, "y": 3}]
                    }
                }

    before = time()
    action_mask = np.array([1, 1, 1, 1]).tolist()

    action = predictor.predict({"state": state, "action_mask": action_mask,
                                "prev_action": -1, 
                               "prev_reward": -1, "seq_lens": -1,  
                               "all_health": health_dict, "json": json})
    elapsed = time() - before

    action_to_take = action["outputs"]["heuristisc_action"]
    print("Action to take {}".format(action_to_take))
    print("Inference took %.2f ms" % (elapsed*1000))