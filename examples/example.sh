#!/bin/bash

# Default values
INPUT_DATASET="TAUR-dev/9_8_25__countdown_3arg__sft_data_multiprompts_reflections"
OUTPUT_DATASET="TAUR-dev/skillfactory_sft_countdown_3arg"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input-dataset)
            INPUT_DATASET="$2"
            shift 2
            ;;
        -o|--output-dataset)
            OUTPUT_DATASET="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

python skill_factory/sft_data_creation.py \
--input_reflection_dataset_name "$INPUT_DATASET" \
--output_dataset_name "$OUTPUT_DATASET" \
--num_responses_ignore_correctness 5 \
--num_repeats_per_question 1 \
--formats '*C_full'