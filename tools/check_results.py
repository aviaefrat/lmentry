import argparse
import pprint
import json
import os


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="CLI for checking results after compare_models.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("ref_results", type=str, help="Reference results")
    parser.add_argument("comp_results", type=str, help="Results for comparison")
    return parser.parse_args()


def main():
    args = parse_arguments()
    ref_data = load_json(args.ref_results)
    comp_data = load_json(args.comp_results)
    process_data(ref_data, comp_data)


def load_json(file_name):
    if not os.path.exists(file_name):
        raise Exception(f"File {file_name} doesn't exists.")
    with open(file_name) as f:
        data = json.load(f)
    return data


def process_data(ref_data, comp_data):
    same_good_dict = {}
    same_bad_dict = {}
    ref_correct = {}
    comp_correct = {}
    rest_dict = {}
    for key in ref_data.keys():
        if not isinstance(ref_data[key], dict):
            continue
        inp = ref_data[key]["input"]
        ref_pred = str(ref_data[key]["prediction"]).replace(inp, "")
        comp_pred = str(comp_data[key]["prediction"]).replace(inp, "")
        if (
            ref_pred == comp_pred
            and ref_data[key]["score"] == 1
            and ref_data[key]["score"] == comp_data[key]["score"]
            and ref_data[key]["certainty"] == comp_data[key]["certainty"]
        ):
            same_good_dict[ref_data[key]["input"]] = {
                "prediction": ref_pred,
                "score": ref_data[key]["score"],
                "certainty": ref_data[key]["certainty"],
            }
        elif (
            ref_pred == comp_pred
            and ref_data[key]["score"] == 0
            and ref_data[key]["score"] == comp_data[key]["score"]
            and ref_data[key]["certainty"] == comp_data[key]["certainty"]
        ):
            same_bad_dict[ref_data[key]["input"]] = {
                "prediction": ref_pred,
                "score": ref_data[key]["score"],
                "certainty": ref_data[key]["certainty"],
            }
        elif (
            ref_data[key]["score"] == 1
            and ref_data[key]["certainty"] == 1
        ):
            ref_correct[ref_data[key]["input"]] = {
                "ref_prediction": ref_pred,
                "comp_prediction": comp_pred,
                "ref_score": ref_data[key]["score"],
                "comp_score": comp_data[key]["score"],
                "ref_certainty": ref_data[key]["certainty"],
                "comp_certainty": comp_data[key]["certainty"],
            }
        elif (
            comp_data[key]["score"] == 1
            and comp_data[key]["certainty"] == 1
        ):
            comp_correct[ref_data[key]["input"]] = {
                "ref_prediction": ref_pred,
                "comp_prediction": comp_pred,
                "ref_score": ref_data[key]["score"],
                "comp_score": comp_data[key]["score"],
                "ref_certainty": ref_data[key]["certainty"],
                "comp_certainty": comp_data[key]["certainty"],
            }
        else:
            rest_dict[ref_data[key]["input"]] = {
                "ref_prediction": ref_pred,
                "comp_prediction": comp_pred,
                "ref_score": ref_data[key]["score"],
                "comp_score": comp_data[key]["score"],
                "ref_certainty": ref_data[key]["certainty"],
                "comp_certainty": comp_data[key]["certainty"],
            }
    print("Full match: ", len(same_good_dict))
    #print(json.dumps(same_good_dict, indent=4, sort_keys=True))
    #print("Correct non match: ", len(same_bad_dict))
    #print(json.dumps(same_bad_dict, indent=4, sort_keys=True))
    print("Ref correct: ", len(ref_correct))
    print(json.dumps(ref_correct, indent=4, sort_keys=True))
    #print("rest: ", len(rest_dict))
    #print(json.dumps(rest_dict, indent=4, sort_keys=True))


if __name__ == "__main__":
    main()
