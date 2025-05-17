import argparse

def run_tool(input_dir, output_dir):
    global input_data, text_list

    input_data = load_input_data(input_dir)
    text_list = list(input_data.values())
    ensure_output_directory(output_dir)

    # Replace fixed 'output' with `output_dir` where needed
    globals()['output_file'] = os.path.join(output_dir, 'semiotic_analysis_output.json')

    # Run full pipeline (reuse all prior functions)
    # You can refactor the script so functions can be re-used here
    # For brevity, insert the rest of your logic from main body
    # Example: analyze_sentiment(text_list[0]), export_to_json(...), etc.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Semiotic Analysis Tool")
    parser.add_argument("--input_dir", type=str, default=".", help="Directory containing input files")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory for saving outputs")
    args = parser.parse_args()

    run_tool(args.input_dir, args.output_dir)
