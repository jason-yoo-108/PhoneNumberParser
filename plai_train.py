import pyprob
from plai import PhoneParser

model = PhoneParser()

model.learn_inference_network(
    inference_network=pyprob.InferenceNetwork.LSTM,
    observe_embeddings={'phone_string': {'dim' : 256}},
    num_traces=5000000,
    batch_size=128,
    save_file_name_prefix="nn_model/phone_parser",
)

model.save_inference_network("nn_model/phone_parser")