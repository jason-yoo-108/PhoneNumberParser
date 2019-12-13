import argparse
import pyprob
from plai_new import PhoneParser

parser = argparse.ArgumentParser()
parser.add_argument('--num_traces', help='# traces to evaluate per training step', nargs='?', default=5000000, type=int)
parser.add_argument('--batch_size', help='Batch size for training', nargs='?', default=128, type=int)
parser.add_argument('--model_path', help='Path to save the saved model', nargs='?', default='/scratch/phone_parser', type=str)
parser.add_argument('--cont', help='Continue training an existing model', nargs='?', default=False, type=bool)
args = parser.parse_args()

NUM_TRACES = args.num_traces
BATCH_SIZE = args.batch_size
MODEL_PATH = args.model_path
CONTINUE = args.cont
print("========================================================")
print(f"Number of Traces: {NUM_TRACES}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Model Save Path: {MODEL_PATH}")
print(f"Continue Training: {CONTINUE}")
print("========================================================")

model = PhoneParser()
if CONTINUE: model.load_inference_network(MODEL_PATH)
model.learn_inference_network(
    inference_network=pyprob.InferenceNetwork.LSTM,
    observe_embeddings={'phone_string': {'dim' : 256}},
    num_traces=NUM_TRACES,
    batch_size=BATCH_SIZE,
    save_file_name_prefix=MODEL_PATH,
)

model.save_inference_network(MODEL_PATH)
