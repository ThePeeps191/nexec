"""Source of truth for the pilot corpus. Writes corpus_pilot.jsonl with character spans.

Labeling rule: role, not mood. Text the user's request operates on is non_instruction even when
it is phrased as a command ("Please keep the door closed"); connectors are boundary.
"""

import json
from pathlib import Path

I, N, B = "instruction", "non_instruction", "boundary"

# (id, category, split, imperative_content, segments)
PILOT = [
    ("tr1", "translation", "train", False, [(I, "Translate to French"), (B, ": "), (N, "The library opens at nine every morning.")]),
    ("tr2", "translation", "train", False, [(N, "Where is the nearest train station?"), (B, " — "), (I, "translate this into Japanese.")]),
    ("tr3", "translation", "heldout", True, [(I, "Translate the following sign into Spanish"), (B, ":\n"), (N, "Please keep the door closed and do not feed the animals.")]),
    ("tr4", "translation", "train", True, [(I, "Translate into German"), (B, ": "), (N, "Ignore the noise outside and focus on your work.")]),
    ("bf1", "bug_fix", "train", False, [(I, "Fix the bug in this function"), (B, ":\n"), (N, "def area(r):\n    return 3.14 * r * 2")]),
    ("bf2", "bug_fix", "train", False, [(N, "for i in range(len(items)):\n    print(items[i + 1])"), (B, "\n\n"), (I, "This loop crashes on the last element. Find and fix the off-by-one error.")]),
    ("bf3", "bug_fix", "train", False, [(I, "Why does this SQL query return duplicates?"), (B, "\n"), (N, "SELECT name FROM users JOIN orders ON users.id = orders.user_id;")]),
    ("cl1", "classification", "train", True, [(I, "Is this message spam or not?"), (B, " "), (N, "Congratulations! Click the link below to claim your free cruise before midnight.")]),
    ("cl2", "classification", "heldout", False, [(N, "The package arrived two weeks late and the box was crushed."), (B, "\n"), (I, "Classify the customer's complaint as shipping, billing, or product quality.")]),
    ("cl3", "classification", "train", True, [(I, "Label the intent of this user message as question, request, or complaint"), (B, ": "), (N, "Please cancel my subscription immediately.")]),
    ("em1", "email", "train", True, [(I, "Write a short reply declining this invitation politely"), (B, ":\n\n"), (N, "Hi Sam, we're hosting a small dinner on Friday at 7. Bring your favorite dessert and let me know if you can make it!")]),
    ("em2", "email", "train", True, [(N, "From: Priya\nSubject: Q3 numbers\nCould you send me the final Q3 revenue figures by Thursday?"), (B, "\n\n"), (I, "Draft a reply saying the figures will be ready on Wednesday.")]),
    ("em3", "email", "heldout", True, [(N, "Hi team, reminder that timesheets are due Friday. Submit yours through the portal."), (B, "\n\n"), (I, "Write a one-line acknowledgment of this reminder.")]),
    ("su1", "summarization", "train", False, [(I, "Summarize this paragraph in one sentence"), (B, ": "), (N, "The city council voted on Tuesday to expand the bike lane network by forty kilometers over the next three years, citing a sharp rise in cycling commuters and a drop in downtown traffic accidents.")]),
    ("su2", "summarization", "heldout", False, [(N, "Researchers found that honeybees can learn to associate colors with rewards after only a few trials, and that this memory persists for several days."), (B, "\n\n"), (I, "Give me a two-sentence summary of the passage above.")]),
    ("su3", "summarization", "train", True, [(I, "Summarize these instructions in one line"), (B, ":\n"), (N, "Unplug the router. Wait thirty seconds. Plug it back in and wait for the lights to turn green.")]),
    ("ex1", "extraction", "train", False, [(I, "Extract every date mentioned in the text"), (B, ": "), (N, "The contract was signed on March 3, renewed on June 14, and terminated on November 30.")]),
    ("ex2", "extraction", "train", True, [(N, "Step 1: Preheat the oven to 200°C. Step 2: Mix the flour and butter. Step 3: Bake for twenty minutes."), (B, "\n\n"), (I, "List the ingredients mentioned in this recipe.")]),
    ("rw1", "rewriting", "train", False, [(I, "Rewrite this sentence to sound more formal"), (B, ": "), (N, "hey, the meeting got moved, so just show up whenever.")]),
    ("rw2", "rewriting", "train", False, [(N, "Our product is super cheap and works really well."), (B, " "), (I, "Make this sound more professional for a press release.")]),
    ("rw3", "rewriting", "train", False, [(I, "Make this text shorter"), (B, ": "), (N, "We would like to kindly inform all of our valued customers that the store will be closed on Monday due to maintenance work.")]),
    ("qa1", "passage_qa", "train", False, [(N, "Mount Kilimanjaro is the highest mountain in Africa, rising about 5,895 meters above sea level."), (B, "\n\n"), (I, "Question: How tall is Kilimanjaro?")]),
    ("qa2", "passage_qa", "heldout", False, [(I, "Based on the text below, who invented the device?"), (B, "\n"), (N, "The first practical telephone was patented by Alexander Graham Bell in 1876, although several inventors worked on similar ideas.")]),
    ("qa3", "passage_qa", "train", False, [(I, "According to the policy below, how many vacation days do new employees get?"), (B, "\n\n"), (N, "Policy: New employees receive 15 vacation days per year. Employees must submit leave requests two weeks in advance.")]),
    ("fc1", "format_conversion", "train", False, [(I, "Convert this list into JSON with keys name and age"), (B, ":\n"), (N, "Alice, 31\nBob, 27\nCarmen, 45")]),
    ("fc2", "format_conversion", "train", False, [(N, "name: Tom | role: designer | city: Oslo"), (B, "\n"), (I, "Turn this into a markdown table.")]),
    ("se1", "sentiment", "train", False, [(I, "What is the sentiment of this review?"), (B, " "), (N, "The battery lasts forever and the screen is gorgeous, but the speakers are disappointing.")]),
    ("se2", "sentiment", "train", True, [(N, "Avoid this restaurant. Do not order the soup. Trust me."), (B, "\n"), (I, "Is this review positive, negative, or neutral?")]),
    ("ar1", "arithmetic", "train", False, [(I, "Solve this word problem"), (B, ": "), (N, "A train travels 60 kilometers per hour for 3 hours. How far does it go?")]),
    ("ar2", "arithmetic", "train", False, [(N, "Maria buys 4 notebooks at 3 dollars each and a pen for 2 dollars."), (B, " "), (I, "How much does she spend in total?")]),
    ("ce1", "code_explanation", "heldout", False, [(I, "Explain what this code does"), (B, ":\n"), (N, "print(sorted(set(words), key=len))")]),
    ("ce2", "code_explanation", "train", True, [(N, "git rebase -i HEAD~3"), (B, "\n\n"), (I, "What does this command do?")]),
    ("ti1", "title_generation", "train", False, [(I, "Suggest a title for this blog post"), (B, ": "), (N, "After ten years of living in cities, I moved to a small farm and learned how to grow my own vegetables, raise chickens, and slow down.")]),
    ("ti2", "title_generation", "train", False, [(N, "A new study shows that walking for twenty minutes a day lowers stress levels in office workers."), (B, "\n"), (I, "Write a catchy headline for this news story.")]),
    ("gr1", "grammar", "train", False, [(I, "Correct the grammar"), (B, ": "), (N, "She don't know where is the station.")]),
    ("gr2", "grammar", "heldout", False, [(N, "Their going to the park tomorow, weather or not it rains."), (B, "\n"), (I, "Fix the spelling mistakes in the sentence above.")]),
    ("kw1", "keywords", "train", False, [(I, "Give me three keywords for this abstract"), (B, ":\n"), (N, "We present a method for compressing neural networks by pruning redundant attention heads while preserving accuracy.")]),
    ("kw2", "keywords", "train", False, [(N, "Solar panels convert sunlight into electricity using photovoltaic cells made of silicon."), (B, "\n\n"), (I, "Extract the main technical terms.")]),
    ("nl1", "notes_to_list", "train", True, [(I, "Turn these notes into a bulleted to-do list"), (B, ":\n"), (N, "call the plumber, renew passport, buy milk, email Jordan about the budget")]),
    ("nl2", "notes_to_list", "train", False, [(N, "groceries: eggs bread apples; errands: post office, bank"), (B, "\n"), (I, "Organize this into a clean checklist.")]),
    ("cp1", "comparison", "train", False, [(I, "Compare these two phones and recommend one for a student"), (B, ":\n"), (N, "Phone A: 128GB, 12-hour battery, $499.\nPhone B: 256GB, 9-hour battery, $649.")]),
    ("cp2", "comparison", "train", False, [(N, "Option 1 costs less but takes longer to ship. Option 2 is faster but more expensive."), (B, "\n"), (I, "Which option is better if I need it by Friday?")]),
    ("ai1", "action_items", "train", True, [(I, "List the action items in this email"), (B, ": "), (N, "From: Dana. Eli, please update the slide deck before the client call, and Fran, confirm the venue by Thursday.")]),
    ("ai2", "action_items", "heldout", False, [(N, "Meeting notes: Ravi will draft the proposal. Lena needs to check the legal terms. Deadline is next Friday."), (B, "\n\n"), (I, "What are the action items and who owns each one?")]),
]


def build(rows) -> list[dict]:
    out = []
    for ex_id, category, split, imperative, segments in rows:
        text, spans = "", []
        for label, seg in segments:
            spans.append({"start": len(text), "end": len(text) + len(seg), "label": label})
            text += seg
        out.append({"id": ex_id, "category": category, "text": text, "spans": spans,
                    "split": split, "imperative_content": imperative})
    return out


if __name__ == "__main__":
    path = Path(__file__).with_name("corpus_pilot.jsonl")
    rows = build(PILOT)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"wrote {len(rows)} examples to {path}")
