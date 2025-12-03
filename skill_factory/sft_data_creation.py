"""
Supported formats
random -> Randomly sample responses from the dataset. Correctness is ignored. Always does "full" range.
no_reflection -> Same as *C_full but no reflections.
0C --> Immediately correct answer. Stop here.
*C --> This is the standard thing we were doing before. N can be 1 to 4.
*C_full --> Same as *C but it goes through the full range from N=1 to N
*C*IC --> After incorrect answer, it goes back to a previous correct answer.
*CC --> After two correct answers, it outputs a special phrase and then the final answer.

Sample call:
python yolo_run_scripts/sft_data_creation_v2.py --num_responses_ignore_correctness 3 --num_repeats_per_question 1 --output_dataset_name TAUR-dev/skillfactory_sft_countdown_3arg --force_unique_prompting_types
"""

from dataclasses import dataclass
from typing import List, Dict
import random
import argparse
from datasets import load_dataset, Dataset

generic_glue_phrases = [
    'However, I should double check this answer.',
    'But wait, let me think about it again.',
    'I can resolve this question to be sure.',
    'Let me verify my answer.',
    'I should check my response again.',
    'I can double check my response.',
    'Wait...',
    'Wait! I should double check my answer.',
    'Although, if I want to be absolutely sure, I should do this again.',
    "I'll recheck what I said earlier.",
    "Time to review my response one more time.",
]

correct_glue_phrases = [
    "This previous answer was correct, but I should double check it to be sure.",
    "Let me try this question again to verify that my response is actually correct.",
    "My earlier answer seems correct, but I should double-check it to be sure.",
    "That response looks right, and I have verified it. It might be worth doinng it again just in case."
    "That answer seems fine, but I'd like to double-check for to be safe.",
    "I believe that was the right answer, but let me make sure.",
    "My previous response looks accurate, though I should recheck it.",
    "The solution seems right. I will now retry it to be more confident.",
    "Looking back, my earlier answer seems right, though I'll recheck it."
    "I'm fairly confident the last answer was right, but I'll double-check anyway."
    "That response looks solid, though I want to be certain.",
    "I'm leaning toward my last answer being right, but I'll test it once more."
    "It's better to be cautious — I'll re-verify my previous answer.",
    "Seems right to me, but a second look won't hurt.",
]

incorrect_glue_phrases = [
    'My previous answer was incorrect. I will now try again.',
    "On review, my last response falls short, so I'll attempt a new one."
    "After reconsideration, I can see my earlier answer wasn't right, and I'll try again.",
    "I learned from my mistake in the last answer — let me rework it.",
    "I may have missed the mark earlier. Let me rethink and attempt again.",
    "Instead of sticking with my incorrect answer, I'll try a new approach.",
    "Oops, I see the issue now — time for another try.",
    "I realize that wasn't the right answer. Let's fix it.",
    "I see the flaw in my earlier response. I'll try a new one.",
    "I made an error before, so I'll reconsider and answer again.",
    "Oops, that wasn't right. Let me take another shot.",
    "Looks like I messed up earlier. I'll go again.",
    "Since my earlier answer was incorrect, I'll rework the reasoning and attempt again.",
    "My last attempt wasn't correct, but I'll refine it and try again.",
]

ending_phrases = [
    "The final answer is:",
    "Therefore, the solution comes out to:",
    "Hence, the correct answer is:",
    "In conclusion, the answer is:",
    "So after checking, the final result is:",
    "That confirms the final answer is:",
    "Altogether, this gives us:",
]

backtrack_phrases = [
    "My current answer is incorrect, but I was able to derive a correct answer in a previous attempt. Let me revisit my previous work.",
    "I can see that this answer is mistaken, but I had worked out the correct one previously. I'll revisit that reasoning.",
    "This one's off, but I did have it right in a past try. I'll look back at that.",
    "While this solution fails, a previous attempt succeeded. I should revisit and adapt that earlier work.",
    "This one's wrong, but I had it right earlier. Let's go back.",
    "This response contains errors, but my earlier calculation was correct. I'll retrace those steps",
    "This response doesn't check out, but I remember arriving at the correct one previously. I'll go back to that",
    "Revisiting my earlier solution makes sense here, as my current answer is incorrect.",
    "I know I got it right in a past attempt, but this time I didn't — better revisit what worked",
]

correct_confirm_phrases = [
    "I have received a correct answer and verified it multiple times. I will now provide the final answer.",
    "This solution has been verified thoroughly and is correct, so I'll share the final answer",
    "Here is my final answer, which I have verified multiple times as correct.",
    "I'm ready to provide the final answer, as I've already confirmed it several times.",
    "Checked repeatedly, confirmed correct: this is my final answer.",
    "I've verified this enough to be sure. Here's my final answer.",
    "I double-checked and it holds up — I'll finalize the answer now.",
]


def generate_dynamic_prompt_template(num_responses):
    prompt_parts = ["<think>"]

    # Add responses
    for i in range(num_responses):
        prompt_parts.append(f"<sample>\n{{response_{i + 1}}}\n</sample>")
        prompt_parts.append(f"<reflect>\n{{reflection_{i + 1}}}\n</reflect>")
        if i < num_responses - 1:  # Don't add retry phrase after the last correct response
            prompt_parts.append(f"{{glue_phrase_{i + 1}}}")

    prompt_parts.append("</think>")
    prompt_parts.append("\n\n{ending_phrase}\n\n<answer>\n{final_answer}\n</answer>")

    return "\n".join(prompt_parts)


@dataclass
class Reflection:
    reflection: str
    verdict: str

    def is_valid(self, response_correctness):
        if self.verdict is None:
            return False
        if self.verdict == response_correctness:
            return True
        return False

    def is_valid_reflection(self):
        if 'idea' in self.reflection.lower():
            print(self.reflection)
            print("----")
            return False
        if 'fast' in self.reflection.lower():
            print(self.reflection)
            print("----")
            return False
        return True


@dataclass
class Response:
    response: str
    response_correctness: bool
    extracted_answer: str
    prompt_variant: str
    reflections: List[Reflection]

    def filter_invalid_reflections(self):
        for refl in self.reflections:
            if "<verdict>" in refl.reflection and "</verdict>" in refl.reflection:
                verdict = refl.reflection.split("<verdict>")[1].split("</verdict>")[0]
                verdict = verdict.lower().strip()
                if verdict == "correct":
                    refl.verdict = True
                elif verdict == "incorrect":
                    refl.verdict = False
                else:
                    refl.verdict = None
        self.reflections = [refl for refl in self.reflections if
                            refl.is_valid(self.response_correctness) and refl.is_valid_reflection()]


@dataclass
class QA_Tuple:
    question: str
    answer: str
    responses: List[Response]
    responses_correct: List[Response] = None
    responses_incorrect: List[Response] = None
    responses_by_prompt_variant: Dict[str, List[Response]] = None

    def sample_responses(self, num_responses, force_unique_prompting_types, finish_with_correct=True):
        if finish_with_correct:
            correct_sample = random.choice(self.responses_correct)
        else:
            correct_sample = random.choice(self.responses)
        remaining_responses = [response for response in self.responses if response != correct_sample]
        if force_unique_prompting_types:
            remaining_prompt_variants = set(list(self.responses_by_prompt_variant.keys()))
            remaining_prompt_variants.remove(correct_sample.prompt_variant)
            selected_prompt_variants = random.sample(list(remaining_prompt_variants), num_responses - 1)
            selected_samples = [random.choice(self.responses_by_prompt_variant[prompt_variant]) for prompt_variant in
                                selected_prompt_variants]
            selected_samples.append(correct_sample)
            return selected_samples
        else:
            selected_samples = random.sample(remaining_responses, num_responses - 1)
            return selected_samples + [correct_sample]

    def sample_backtrack_to_correct(self, num_responses):
        correct_sample = random.choice(self.responses_correct)
        incorrect_sample = random.choice(self.responses_incorrect)
        remaining_responses = [response for response in self.responses if
                               response != correct_sample and response != incorrect_sample]
        selected_samples = [correct_sample] + random.sample(remaining_responses, num_responses - 2)
        random.shuffle(selected_samples)
        selected_samples = selected_samples + [incorrect_sample]
        return selected_samples

    def sample_correct_confirm(self, num_responses):
        two_correct_samples = random.sample(self.responses_correct, 2)
        remaining_responses = [response for response in self.responses if
                               response != two_correct_samples[0] and response != two_correct_samples[1]]
        selected_samples = random.sample(remaining_responses, num_responses - 2)
        selected_samples = selected_samples + two_correct_samples
        return selected_samples


parser = argparse.ArgumentParser()
parser.add_argument("--input_reflection_dataset_name", type=str,
                    default="TAUR-dev/9_8_25__countdown_3arg__sft_data_multiprompts_reflections")
parser.add_argument("--output_dataset_name", type=str, default="TAUR-dev/skillfactory_sft_countdown_3arg")
parser.add_argument("--num_responses_ignore_correctness", type=int, default=None)
parser.add_argument("--subsample_size", type=int, default=None)
parser.add_argument("--num_repeats_per_question", type=int, default=3)
parser.add_argument("--sampling_response_counts", action="store_true")

parser.add_argument("--formats", nargs='+', default=["0C", "*C", "*C*IC", "*CC"])
parser.add_argument("--force_unique_prompting_types", action="store_true")
parser.add_argument("--prompt_variants_to_use", nargs='+',
                    default=["original", "plan_and_execute", "alternatively", "rephrase"])

parser.add_argument("--response_column", type=str, default="model_responses__mutated_prompts")
parser.add_argument("--extracted_answer_column", type=str,
                    default="model_responses__mutated_prompts__eval_extracted_answers")
parser.add_argument("--responses_correctness_column", type=str,
                    default="model_responses__mutated_prompts__eval_is_correct")
parser.add_argument("--reflection_column", type=str, default="model_responses__mutated_prompts_reflection")
args = parser.parse_args()

reflection_dataset = load_dataset(args.input_reflection_dataset_name)['train']

# Filter out prompt variants not in args.prompt_variants_to_use
reflection_dataset = reflection_dataset.filter(lambda x: x["prompt_variant"] in args.prompt_variants_to_use)

final_dataset = {"conversations": [], "sft_template_type_idx": []}

grouped_reflection_dataset = reflection_dataset.to_pandas().groupby("question")

responses_valid = []
for question, group in grouped_reflection_dataset:
    valid_responses_per_question = []

    for responses, responses_correctness, extracted_answer, reflection_list, prompt_variant in zip(
            group[args.response_column],
            group[args.responses_correctness_column],
            group[args.extracted_answer_column],
            group[args.reflection_column],
            group["prompt_variant"]
    ):
        response = Response(
            response=responses[0],
            response_correctness=responses_correctness[0],
            extracted_answer=extracted_answer[0],
            prompt_variant=prompt_variant,
            reflections=[Reflection(reflection=refl, verdict=None) for refl in reflection_list],
        )
        response.filter_invalid_reflections()
        if len(response.reflections) > 0:
            valid_responses_per_question.append(response)
    qa = QA_Tuple(
        question=question,
        answer=group["answer"].tolist()[0],
        responses=valid_responses_per_question
    )
    qa.responses_correct = [response for response in qa.responses if response.response_correctness == True]
    qa.responses_incorrect = [response for response in qa.responses if response.response_correctness == False]
    qa.responses_by_prompt_variant = {
        prompt_variant: [response for response in qa.responses if response.prompt_variant == prompt_variant]
        for prompt_variant in set([response.prompt_variant for response in qa.responses])
    }
    if len(qa.responses) >= args.num_responses_ignore_correctness:
        responses_valid.append(qa)

print("len(responses_valid)", len(responses_valid))

for qa in responses_valid:
    if len(qa.responses_correct) == 0:
        continue
    if args.force_unique_prompting_types:
        if len(qa.responses_by_prompt_variant) < args.num_responses_ignore_correctness:
            continue
    for i in range(args.num_repeats_per_question):
        if "random" in args.formats:
            for N in range(1, args.num_responses_ignore_correctness + 1):
                prompt_template = generate_dynamic_prompt_template(N)
                selected_responses = qa.sample_responses(N, args.force_unique_prompting_types,
                                                         finish_with_correct=False)
                responses_data = {"final_answer": selected_responses[-1].extracted_answer,
                                  "ending_phrase": random.choice(ending_phrases)}
                for i in range(N):
                    responses_data[f"response_{i + 1}"] = selected_responses[i].response
                    responses_data[f"reflection_{i + 1}"] = random.choice(selected_responses[i].reflections).reflection
                    if i < N - 1:
                        if selected_responses[i].response_correctness == True:
                            responses_data[f"glue_phrase_{i + 1}"] = random.choice(
                                correct_glue_phrases + generic_glue_phrases)
                        else:
                            responses_data[f"glue_phrase_{i + 1}"] = random.choice(
                                incorrect_glue_phrases + generic_glue_phrases)
                prompt = prompt_template.format(**responses_data)
                final_dataset["conversations"].append([
                    {"role": "user", "content": qa.question},
                    {"role": "assistant", "content": prompt}
                ])
                final_dataset["sft_template_type_idx"].append("random")
        if "no_reflection" in args.formats:
            for N in range(1, args.num_responses_ignore_correctness + 1):
                prompt_template = generate_dynamic_prompt_template(N)
                selected_responses = qa.sample_responses(N, args.force_unique_prompting_types)
                responses_data = {"final_answer": selected_responses[-1].extracted_answer,
                                  "ending_phrase": random.choice(ending_phrases)}
                for i in range(N):
                    responses_data[f"response_{i + 1}"] = selected_responses[i].response
                    responses_data[f"reflection_{i + 1}"] = ""
                    if i < N - 1:
                        responses_data[f"glue_phrase_{i + 1}"] = random.choice(generic_glue_phrases)
                prompt = prompt_template.format(**responses_data)
                prompt = prompt.replace("<reflect>\n\n</reflect>", "")
                final_dataset["conversations"].append([
                    {"role": "user", "content": qa.question},
                    {"role": "assistant", "content": prompt}
                ])
                final_dataset["sft_template_type_idx"].append("no_reflection")

        if "0C" in args.formats:
            prompt_template = generate_dynamic_prompt_template(1)
            selected_responses = qa.sample_responses(1, args.force_unique_prompting_types)
            responses_data = {"final_answer": selected_responses[-1].extracted_answer,
                              "ending_phrase": random.choice(ending_phrases)}
            for i in range(1):
                responses_data[f"response_{i + 1}"] = selected_responses[i].response
                responses_data[f"reflection_{i + 1}"] = random.choice(selected_responses[i].reflections).reflection
            prompt = prompt_template.format(**responses_data)
            final_dataset["conversations"].append([
                {"role": "user", "content": qa.question},
                {"role": "assistant", "content": prompt}
            ])
            final_dataset["sft_template_type_idx"].append("0C")
        if "*C" in args.formats:
            if args.sampling_response_counts:
                N = random.randint(2, args.num_responses_ignore_correctness)
            else:
                N = args.num_responses_ignore_correctness
            prompt_template = generate_dynamic_prompt_template(N)
            selected_responses = qa.sample_responses(N, args.force_unique_prompting_types)
            responses_data = {"final_answer": selected_responses[-1].extracted_answer,
                              "ending_phrase": random.choice(ending_phrases)}
            for i in range(N):
                responses_data[f"response_{i + 1}"] = selected_responses[i].response
                responses_data[f"reflection_{i + 1}"] = random.choice(selected_responses[i].reflections).reflection
                if i < N - 1:
                    if selected_responses[i].response_correctness == True:
                        responses_data[f"glue_phrase_{i + 1}"] = random.choice(
                            correct_glue_phrases + generic_glue_phrases)
                    else:
                        responses_data[f"glue_phrase_{i + 1}"] = random.choice(
                            incorrect_glue_phrases + generic_glue_phrases)
            prompt = prompt_template.format(**responses_data)
            final_dataset["conversations"].append([
                {"role": "user", "content": qa.question},
                {"role": "assistant", "content": prompt}
            ])
            final_dataset["sft_template_type_idx"].append(f"{N}C")
        if "*C_full" in args.formats:
            for N in range(1, args.num_responses_ignore_correctness + 1):
                prompt_template = generate_dynamic_prompt_template(N)
                selected_responses = qa.sample_responses(N, args.force_unique_prompting_types)
                responses_data = {"final_answer": selected_responses[-1].extracted_answer,
                                  "ending_phrase": random.choice(ending_phrases)}
                for i in range(N):
                    responses_data[f"response_{i + 1}"] = selected_responses[i].response
                    responses_data[f"reflection_{i + 1}"] = random.choice(selected_responses[i].reflections).reflection
                    if i < N - 1:
                        if selected_responses[i].response_correctness == True:
                            responses_data[f"glue_phrase_{i + 1}"] = random.choice(
                                correct_glue_phrases + generic_glue_phrases)
                        else:
                            responses_data[f"glue_phrase_{i + 1}"] = random.choice(
                                incorrect_glue_phrases + generic_glue_phrases)
                prompt = prompt_template.format(**responses_data)
                final_dataset["conversations"].append([
                    {"role": "user", "content": qa.question},
                    {"role": "assistant", "content": prompt}
                ])
                final_dataset["sft_template_type_idx"].append(f"{N}C")
        if "*C*IC" in args.formats:
            if args.sampling_response_counts:
                N = random.randint(2, args.num_responses_ignore_correctness)
            else:
                N = args.num_responses_ignore_correctness
            prompt_template = generate_dynamic_prompt_template(N)
            selected_responses = qa.sample_backtrack_to_correct(N)
            correct_sample_idx = None
            for i in range(N - 1, -1, -1):
                if selected_responses[i].response_correctness == True:
                    correct_sample_idx = i
                    break
            responses_data = {"final_answer": selected_responses[correct_sample_idx].extracted_answer,
                              "ending_phrase": random.choice(ending_phrases)}
            for i in range(N):
                responses_data[f"response_{i + 1}"] = selected_responses[i].response
                responses_data[f"reflection_{i + 1}"] = random.choice(selected_responses[i].reflections).reflection
                if i < N - 1:
                    if selected_responses[i].response_correctness == True:
                        responses_data[f"glue_phrase_{i + 1}"] = random.choice(
                            correct_glue_phrases + generic_glue_phrases)
                    else:
                        responses_data[f"glue_phrase_{i + 1}"] = random.choice(
                            incorrect_glue_phrases + generic_glue_phrases)
            responses_data["ending_phrase"] = random.choice(backtrack_phrases) + '\n' + selected_responses[
                correct_sample_idx].response + '\n' + responses_data["ending_phrase"]
            prompt = prompt_template.format(**responses_data)
            final_dataset["conversations"].append([
                {"role": "user", "content": qa.question},
                {"role": "assistant", "content": prompt}
            ])
            final_dataset["sft_template_type_idx"].append("*C*IC")
        if "*CC" in args.formats and len(qa.responses_correct) >= 2:
            if args.sampling_response_counts:
                N = random.randint(2, args.num_responses_ignore_correctness)
            else:
                N = args.num_responses_ignore_correctness
            prompt_template = generate_dynamic_prompt_template(N)
            selected_responses = qa.sample_correct_confirm(N)
            responses_data = {"final_answer": selected_responses[-1].extracted_answer,
                              "ending_phrase": random.choice(ending_phrases)}
            for i in range(N):
                responses_data[f"response_{i + 1}"] = selected_responses[i].response
                responses_data[f"reflection_{i + 1}"] = random.choice(selected_responses[i].reflections).reflection
                if i < N - 1:
                    if selected_responses[i].response_correctness == True:
                        responses_data[f"glue_phrase_{i + 1}"] = random.choice(
                            correct_glue_phrases + generic_glue_phrases)
                    else:
                        responses_data[f"glue_phrase_{i + 1}"] = random.choice(
                            incorrect_glue_phrases + generic_glue_phrases)
            responses_data["ending_phrase"] = random.choice(correct_confirm_phrases) + '\n' + responses_data[
                "ending_phrase"]
            prompt = prompt_template.format(**responses_data)
            final_dataset["conversations"].append([
                {"role": "user", "content": qa.question},
                {"role": "assistant", "content": prompt}
            ])
            final_dataset["sft_template_type_idx"].append("*CC")

if args.subsample_size is not None:
    final_dataset["conversations"] = random.sample(final_dataset["conversations"], args.subsample_size)
ds = Dataset.from_dict(final_dataset)
# output_dataset_name = f"{args.output_dataset_name}_qrepeat{args.num_repeats_per_question}_reflections{args.num_responses_ignore_correctness}_formats{('.'.join(args.formats)).replace('*', '-')}"
output_dataset_name = f"{args.output_dataset_name}_reflections{args.num_responses_ignore_correctness}_formats{('.'.join(args.formats)).replace('*', '-')}"
if len(output_dataset_name) > 95:
    print("Output dataset name is too long, truncating...")
    output_dataset_name = output_dataset_name[:95].rstrip('.')
print("Output dataset name:", output_dataset_name)
print("Output dataset:", ds)
ds.shuffle(seed=42)
ds.push_to_hub(output_dataset_name)