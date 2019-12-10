import pyprob
import sys
from plai import PhoneParser

model = PhoneParser()
TARGET = sys.argv[1] if len(sys.argv)>1 else '+1 (604) 250 1363'

model.load_inference_network('nn_model/phone_parser')
post = model.posterior_distribution(
    observe=model.get_observes(TARGET),
    inference_engine=pyprob.InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK,
    num_traces=10
)

for i in range(10):
    print(post.sample())
