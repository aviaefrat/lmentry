import argparse
import pprint
import json
import os
import re


SAVE_RESULTS = ["files", "html"]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="CLI for checking results after compare_models.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("pred_results", type=str, help="Prediction results")
    parser.add_argument(
        "--save_results",
        type=str,
        default=SAVE_RESULTS[0],
        help="Write results to files or HTML",
        choices=SAVE_RESULTS,
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    ref_data = load_json(args.pred_results)
    res_tuple = process_data(ref_data)
    if args.save_results == "files":
        (
            correct_cert,
            correct_non_cert,
            non_correct_non_cert,
            non_correct_cert,
        ) = res_tuple
        write_to_file(correct_cert, "Correct and certainty", "correct_cert.txt")
        write_to_file(
            correct_non_cert, "Correct and non certainty", "correct_non_cert.txt"
        )
        write_to_file(
            non_correct_non_cert,
            "Not correct and non certainty",
            "non_correct_non_cert.txt",
        )
        write_to_file(
            non_correct_cert, "Not correct and certainty", "non_correct_cert.txt"
        )
    else:
        title = "Results for {}".format(args.pred_results)
        generate_html(title, res_tuple, "output.html")


def load_json(file_name):
    if not os.path.exists(file_name):
        raise Exception(f"File {file_name} doesn't exists.")
    with open(file_name) as f:
        data = json.load(f)
    return data


def write_to_file(data, title, file_name):
    with open(file_name, "w") as f:
        f.write(title + ": " + str(len(data)) + "\n")
        f.write(json.dumps(data, indent=4, sort_keys=True))


def process_data(data):
    correct_cert = {}
    correct_non_cert = {}
    non_correct_non_cert = {}
    non_correct_cert = {}
    for key in data.keys():
        if not isinstance(data[key], dict):
            continue
        inp = data[key]["input"].strip('\n\r ')
        pred = str(data[key]["prediction"]).replace(inp, "")
        score = data[key]["score"]
        cert = data[key]["certainty"]
        if score == 1 and cert == 1:
            correct_cert[inp] = {
                "prediction": pred,
                "score": score,
                "certainty": cert,
            }
        elif score == 1 and cert == 0:
            correct_non_cert[inp] = {
                "prediction": pred,
                "score": score,
                "certainty": cert,
            }
        elif score == 0 and cert == 0:
            non_correct_non_cert[inp] = {
                "prediction": pred,
                "score": score,
                "certainty": cert,
            }
        else:
            non_correct_cert[inp] = {
                "prediction": pred,
                "score": score,
                "certainty": cert,
            }
    return correct_cert, correct_non_cert, non_correct_non_cert, non_correct_cert


def generate_html(title, res_tuple, file_name):
    def _header(title):
        head = f"""
        <!doctype html>
        <html lang="en-US">
          <head>
            <meta charset="utf-8" />
            <title>{title}</title>
          </head>
          <body>
        """
        return head

    def _footer():
        return """
          </body>
        </html>
        """

    def _add_results(title, results):
        str = f"""
            <details>
            <summary><strong><font size="5">{title}</font></strong></summary>
            <br>
        """
        for inp, res in results.items():
            str += """
            <b>Input:</b> <i>{}</i><br>
            <b>Prediction:</b> <i>{}</i><br>
            <u>Score: {}. Certainty: {}</u><br>
            <br>
            """.format(inp, res["prediction"], res["score"], res["certainty"])
        str += """
            </details>
        """
        return str

    str = _header(title)
    (
        correct_cert,
        correct_non_cert,
        non_correct_non_cert,
        non_correct_cert,
    ) = res_tuple
    str += _add_results("Correct and certainty", correct_cert)
    str += _add_results("Correct and non certainty", correct_non_cert)
    str += _add_results("Not correct and non certainty", non_correct_non_cert)
    str += _add_results("Not correct and certainty", non_correct_cert)
    str += _footer()

    with open(file_name, "w") as f:
        f.write(str)


if __name__ == "__main__":
    main()
