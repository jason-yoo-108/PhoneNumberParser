import argparse
import pyprob
from plai_new import PhoneParser

parser = argparse.ArgumentParser()
parser.add_argument('--number', help='Number to parse', nargs='?', default='+1 (604) 250 1363', type=str)
parser.add_argument('--num_traces', help='# traces to evaluate during inference', nargs='?', default=10, type=int)
parser.add_argument('--num_samples', help='# samples to sample from the posterior', nargs='?', default=10, type=int)
parser.add_argument('--model_path', help='Path to the saved model', nargs='?', default='/scratch/phone_parser', type=str)
args = parser.parse_args()

NUMBER = args.number
NUM_TRACES = args.num_traces
NUM_SAMPLES = args.num_samples
MODEL_PATH = args.model_path
print("========================================================")
print(f"Phone Number to Parse: {NUMBER}")
print(f"Number of Traces / Samples: {NUM_TRACES}, {NUM_SAMPLES}")
print(f"Model Path: {MODEL_PATH}")
print("========================================================")


model = PhoneParser()
model.load_inference_network(MODEL_PATH)
post = model.posterior_distribution(
    observe=model.get_observes(NUMBER),
    inference_engine=pyprob.InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK,
    num_traces=NUM_TRACES
)

for i in range(NUM_SAMPLES):
    print(post.sample())
