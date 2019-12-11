import argparse
import pyprob
from plai_new import PhoneParser

parser = argparse.ArgumentParser()
parser.add_argument('--num_traces', help='# traces to evaluate per training step', nargs='?', default=5000000, type=int)
parser.add_argument('--batch_size', help='Batch size for training', nargs='?', default=128, type=int)
parser.add_argument('--model_path', help='Path to save the saved model', nargs='?', default='/scratch/phone_parser', type=str)
args = parser.parse_args()

NUM_TRACES = args.num_traces
BATCH_SIZE = args.batch_size
MODEL_PATH = args.model_path
print("========================================================")
print(f"Number of Traces: {NUM_TRACES}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Model Save Path: {MODEL_PATH}")
print("========================================================")

model = PhoneParser()
model.learn_inference_network(
    inference_network=pyprob.InferenceNetwork.LSTM,
    observe_embeddings={'phone_string': {'dim' : 256}},
    num_traces=NUM_TRACES,
    batch_size=BATCH_SIZE,
    save_file_name_prefix=MODEL_PATH,
)

model.save_inference_network(MODEL_PATH)