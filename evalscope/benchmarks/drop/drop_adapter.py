import re
from typing import List

from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.constants import EvalType
from evalscope.utils.logger import get_logger

logger = get_logger()

DROP_EXAMPLES = '''Some examples of passages and Q&A are provided below.

# Examples
---
Passage: Trunajaya rebellion  or Trunajaya War was the ultimately unsuccessful rebellion waged by the Madurese prince Trunajaya and fighters from Makassar against the Mataram Sultanate and its Dutch East India Company  supporters in Java  during the 1670s. The rebellion was initially successful: the rebels defeated the royal army at Gegodog , captured most of the Javanese north coast, and took the Mataram capital Plered . King Amangkurat I died during the retreat of the royal court. His son and successor, Amangkurat II, requested help from the VOC in exchange for financial remuneration and geopolitical concessions. The VOC\'s subsequent involvement turned the tide of the war. VOC and Mataram forces recovered lost territories and overran Trunajaya\'s new capital at Kediri . However, the rebellion continued until the capture of Trunajaya at the end of 1679, and the defeat, death, or surrender of the other rebel leaders . Trunajaya was killed by Amangkurat II personally in 1680 while a prisoner of the VOC. After his father\'s death in 1677, Amangkurat II also faced rival claims to the throne. The most serious rival was his brother Pangeran Puger, who took the capital Plered in 1677 and did not surrender until 1681.
Question: How many years was it between Trunajaya\'s capture and his death while prisoner of the VOC?
Answer:  1

---
Passage: Led by former Giant Kurt Warner, the defending NFC champions took the field at Giants Stadium against a Giants team still reeling from their bad loss in New Orleans. The Giants scored first, sending Jacobs in for a 4-yard touchdown run following a Terrell Thomas interception. Later, Arizona running back Beanie Wells scored his first career touchdown on a 13-yard rush. Manning responded by throwing a 62-yard touchdown to Nicks for his longest reception of the year. In the second half, the Cardinals\' Tim Hightower and Jason Wright scored touchdowns. But it was turnovers that decided this game; Manning\'s 3 interceptions were as many as he had thrown all season. The Giants scored only 3 points in the second half, ending the game on an interception to Antrel Rolle. The Giants notable streak of 38 consecutive starts by the same offensive line unit was ended here, as offensive tackle Kareem McKenzie missed the game with a groin injury. McKenzie returned the following week.
Question: Which player made the first score of the game?
Answer:  Jacobs

---
Passage: Hoping to rebound from their road loss to the Bills, the Chargers flew to Wembley Stadium for the 2008 International Series game with the New Orleans Saints. In the first quarter, San Diego trailed early as kicker Taylor Mehlhaff got a 23-yard field goal.  The \'Bolts would respond with kicker Nate Kaeding getting a 33-yard field goal.  In the second quarter, New Orleans regained the lead as QB Drew Brees (a former Charger) completed a 12-yard TD pass to WR Devery Henderson (with a failed PAT) and RB Deuce McAllister getting a 1-yard TD run.  San Diego answered as QB Philip Rivers completed a 12-yard TD pass to RB LaDainian Tomlinson, but the Saints replied with Brees completing a 30-yard TD pass to WR Lance Moore.  The Chargers closed out the half with Rivers completing a 12-yard TD pass to TE Antonio Gates. In the third quarter, New Orleans increased its lead Brees completing a 1-yard TD pass to TE Mark Campbell, after a very controversial Pass interference call on cornerback Cletis Gordon put the Saints on the 1-yard line.  The \'Bolts would answer with Kaeding getting a 24-yard field goal.  In the fourth quarter, the Saints continued to build its lead as FB Mike Karney got a 1-yard TD run.  San Diego tried to rally as Kaeding nailed a 31-yard field goal, Rivers completed a 14-yard TD pass to WR Vincent Jackson, and Brees giving the \'Bolts a safety via an incomplete pass thrown into the back of his own endzone.  However, New Orleans\' defense stiffened for the win. With the loss, the Chargers went into their bye week at 3-5.
Question: How many total yards of touchdown passes did Drew Brees make?
Answer:  43

'''  # noqa: E501


@Benchmark.register(
    name='drop',
    pretty_name='DROP',
    dataset_id='AI-ModelScope/DROP',
    metric_list=['AverageAccuracy'],
    few_shot_num=0,
    train_split=None,
    eval_split='validation',
    prompt_template=
    'You will be asked to read a passage and answer a question.{drop_examples}# Your Task\n\n---\n{query}\n\nThink step by step, then write a line of the form "Answer: $ANSWER" at the end of your response.',  # noqa: E501
)
class DROPAdapter(DataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        few_shot_num = kwargs.get('few_shot_num', 0)
        if few_shot_num != 0:
            self.few_shot_num = 3
            logger.info(f'Few shot num is set to {self.few_shot_num} for DROP dataset by system.')
        else:
            self.few_shot_num = 0

    def gen_prompt(self, input_d: dict, subset_name: str, few_shot_list: list, **kwargs) -> dict:
        """
        Generate model prompt from input data.
        """
        drop_examples = '' if self.few_shot_num == 0 else DROP_EXAMPLES
        query = f"Passage: {input_d['passage']}\nQuestion: {input_d['question']}"
        prompt = self.prompt_template.format(
            drop_examples=drop_examples,
            query=query,
        )
        return self.gen_prompt_data(prompt)

    def get_gold_answer(self, input_d: dict) -> List[str]:
        """
        Parse the raw input labels (gold).
        """

        def _flatten_validated_answers(validated_answers):
            """Flattens a dict of lists of validated answers.
            {"number": ['1', '8'], ...}
            -> [{"number": ['1'], ...}, {"number": ['8'], ...}]
            """
            valid_answers = []
            for i in range(len(validated_answers['number'])):
                valid_answers.append({
                    'number': validated_answers['number'][i],
                    'date': validated_answers['date'][i],
                    'spans': validated_answers['spans'][i],
                })
            return valid_answers

        answers = []
        answers_set = set()
        candidates = [input_d['answer']] + _flatten_validated_answers(input_d['validated_answers'])
        for candidate in candidates:
            answer = DROPAdapter.parse_answer(candidate)
            if answer in answers_set:
                continue
            answers_set.add(answer)
            answers.append(answer)
        return answers

    def parse_pred_result(self, result: str, raw_input_d: dict = None, eval_type: str = EvalType.CHECKPOINT) -> str:
        """
        Parse the predicted result and extract proper answer.
        """
        match = re.search(r'(?i)Answer\s*:\s*([^\n]+)', result)
        extracted_answer = match.group(1) if match else result
        return extracted_answer

    def match(self, gold: List[str], pred: str) -> float:
        """
        Match the gold answer and the predicted answer.
        """
        from .utils import _answer_to_bags

        max_em = 0
        for gold_answer in gold:
            # Convert the answers to bags of answers
            predicted_bags = _answer_to_bags(pred)
            gold_bags = _answer_to_bags(gold_answer)

            if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(gold_bags[0]):
                exact_match = 1.0
            else:
                exact_match = 0.0
            # Check if the answer is empty
            if gold_answer[0].strip():
                max_em = max(max_em, exact_match)

        return max_em

    @staticmethod
    def parse_answer(answer):
        # NOTE: Everything is returned as a tuple for uniformity and hashability.
        if answer['number'] != '':
            return (str(answer['number']), )
        if answer['spans'] != []:
            return tuple(answer['spans'])
        return (' '.join([answer['date']['day'], answer['date']['month'], answer['date']['year']]).strip(), )
