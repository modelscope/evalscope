import json

from evalscope.metrics import compute_rouge_score_one_sample


def evaluate_rougel(cand_list: list, ref_list: list):
    if len(ref_list) == 0:
        return 0
    rouge_score = compute_rouge_score_one_sample(cand_list, ref_list)
    rougel = rouge_score.get('rouge-l-f', 0)

    return rougel


def evaluate_action_em(cand_list: list, ref_list: list):
    if len(ref_list) == 0:
        return 0
    em = 0
    for cand, ref in zip(cand_list, ref_list):
        em += (1 if cand == ref else 0)
    return em / len(cand_list)


def evaluate_action_input_f1(action_pred: list, action_ref: list, cand_list: list, ref_list: list):
    easy_f1 = []
    hard_f1 = []
    f1 = []
    for i in range(len(action_pred)):
        ref_action = action_ref[i]
        pred_action = action_pred[i]

        ref_input = ref_list[i]
        cand_input = cand_list[i]

        if ref_action != pred_action:
            easy_f1.append(0)
            hard_f1.append(0)
            f1.append(0)
        else:
            try:
                ref_input_json = json.loads(ref_input)
                try:
                    cand_input_json = json.loads(cand_input)
                    half_match = 0
                    full_match = 0
                    if ref_input_json == {}:
                        if cand_input_json == {}:
                            easy_f1.append(1)
                            f1.append(1)
                        else:
                            easy_f1.append(0)
                            f1.append(0)
                    else:
                        for k, v in ref_input_json.items():
                            if k in cand_input_json.keys():
                                if cand_input_json[k] == v:
                                    full_match += 1
                                else:
                                    half_match += 1

                        recall = (0.5 * half_match + full_match) / (len(ref_input_json) + 1e-30)
                        precision = (0.5 * half_match + full_match) / (len(cand_input_json) + 1e-30)
                        hard_f1.append((2 * recall * precision) / (recall + precision))
                        f1.append((2 * recall * precision) / (recall + precision))
                except Exception:
                    # cand_input = cand_input.replace("\n","").replace("\"","")
                    # ref_input = cand_input.replace("\n","").replace("\"","")
                    # rouge = Rouge()
                    # rouge_score = rouge.get_scores(hyps=[cand_input], refs=[ref_input], avg=True)
                    if ref_input_json == {}:
                        easy_f1.append(0)
                    else:
                        hard_f1.append(0)
                    # hard_f1.append(rouge_score["rouge-l"]["f"])
                    # f1.append(rouge_score["rouge-l"]["f"])
                    f1.append(0)
            except Exception:
                pass

    # 检查列表是否为空，如果为空则返回0
    easy_f1_avg = sum(easy_f1) / len(easy_f1) if easy_f1 else 0
    hard_f1_avg = sum(hard_f1) / len(hard_f1) if hard_f1 else 0
    f1_avg = sum(f1) / len(f1) if f1 else 0

    return easy_f1_avg, hard_f1_avg, f1_avg


def parse_action(text):
    action = 'None'
    action_input = '{}'
    if 'Action Input:' in text:
        input_idx = text.rindex('Action Input:')
        action_input = text[input_idx + len('Action Input:'):].strip()
    else:
        action_input = '{}'

    if 'Action:' in text:
        action_idx = text.rindex('Action:')
        action = text[action_idx + len('Action:'):].strip()
        if 'Action Input:' in action:
            input_idx = action.index('Action Input:')
            action = action[:input_idx].strip()
    else:
        action = 'none'
    return action, action_input


def parse_output(text):
    action, action_input = parse_action(text)
    if action == 'Finish':
        try:
            action_input = json.loads(action_input)
            # print(action_input)
            # print(json.dumps(action_input,indent=2))
            return_type = action_input['return_type']
            if return_type == 'give_answer':
                if 'final_answer' in action_input.keys():
                    answer = str(action_input['final_answer'])
                    if answer.strip() in ['', '.', ',']:
                        answer = 'None'
                else:
                    answer = 'None'
                return 'finish', action, action_input, answer
            else:
                return 'give up', None, None, None
        except Exception:
            return 'give up', None, None, None
    else:
        plan = 'call'
        answer = None
        return plan, action, action_input, answer


def calculate_metrics(data):
    """
    Calculate the metrics for the given data.
    """
    plan_ref = []
    plan_pred = []
    hallu_cases = []
    answer_ref = []
    action_ref = []
    action_input_ref = []
    answer_pred = []
    action_pred = []
    action_input_pred = []
    hallu_pred = 0

    reference = data['target']
    prediction = data['predictions']
    ref_plan, ref_action, ref_input, ref_ans = parse_output(reference)
    # ref_plan: call
    # ref_action: spott
    # ref_input: {"is_id": "city center" }
    # ref_ans: None

    pred_plan, pred_action, pred_input, pred_ans = parse_output(prediction)
    if ref_action is not None and ref_action == 'invalid_hallucination_function_name':
        return {}
    if pred_action is not None and ref_action != 'none' and ref_action not in [t['name'] for t in data['tools']]:
        return {}

    if pred_action is not None and pred_action != 'none' and pred_action not in [t['name'] for t in data['tools']]:
        hallu_pred += 1
        hallu_cases.append(data)

    plan_ref.append(ref_plan)
    plan_pred.append(pred_plan)
    if ref_plan == 'give up':
        pass
    elif ref_plan == 'finish':
        answer_ref.append(ref_ans)
        if pred_ans is None:
            answer_pred.append('none')
        else:
            answer_pred.append(pred_ans)
    else:
        action_ref.append(ref_action)
        action_input_ref.append(ref_input)
        if pred_action is None:
            action_pred.append('none')
        else:
            action_pred.append(pred_action)

        if pred_input is None:
            action_input_pred.append('{}')
        else:
            action_input_pred.append(pred_input)

    metric = {}
    rouge = evaluate_rougel(answer_pred, answer_ref)
    plan_em = evaluate_action_em(cand_list=plan_pred, ref_list=plan_ref)
    action_em = evaluate_action_em(cand_list=action_pred, ref_list=action_ref)
    easy_f1, hard_f1, f1 = evaluate_action_input_f1(action_pred, action_ref, action_input_pred, action_input_ref)
    hallu_rate = hallu_pred
    metric['Act.EM'] = action_em
    metric['F1'] = f1
    metric['HalluRate'] = hallu_rate
    metric['plan_em'] = plan_em
    metric['Easy_F1'] = easy_f1
    metric['Hard_F1'] = hard_f1
    metric['Rouge-L'] = rouge
    return metric
