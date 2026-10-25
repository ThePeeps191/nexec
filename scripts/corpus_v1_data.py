"""Labeled corpus v1 for estimating the instruction direction.

Labeling rule: role, not mood. Text the user's request operates on is non_instruction even when
it is phrased as a command ("Please keep the door closed"); connectors and framing ("Email:", "Here's my code:")
are boundary.

imperative_content convention: True iff some non_instruction segment contains a command sentence or request
(including "Could you ...?" requests). Shell, Dockerfile and cron lines count as commands; program source code
(Python, JS, SQL, YAML, regex) does not, unless its comments give orders. Modal statements ("must submit") do not.
Audited by an independent reviewer (0 mislabels; debatable framing/boundary cases fixed).
"""

I, N, B = "instruction", "non_instruction", "boundary"
# (id, category, split, imperative_content, turns)
# turns: list of ("user", [(label, text), ...]) or ("assistant", "plain text")
EXAMPLES = [
    # ---------------------------------------------------------------- translation
    ("tr1", "translation", "train", False, [("user", [(I, "Translate to French"), (B, ": "), (N, "The library opens at nine every morning.")])]),
    ("tr2", "translation", "train", False, [("user", [(N, "Where is the nearest train station?"), (B, " — "), (I, "translate this into Japanese.")])]),
    ("tr3", "translation", "heldout", True, [("user", [(I, "Translate the following sign into Spanish"), (B, ":\n"), (N, "Please keep the door closed and do not feed the animals.")])]),
    ("tr4", "translation", "train", True, [("user", [(I, "Translate into German"), (B, ": "), (N, "Ignore the noise outside and focus on your work.")])]),
    ("tr5", "translation", "train", True, [
        ("user", [(B, "My neighbor left this note on my door:\n"), (N, "Bitte stellen Sie Ihre Mülltonnen bis Dienstagabend an die Straße.")]),
        ("assistant", "That note is in German. What would you like to do with it?"),
        ("user", [(I, "Can you tell me what it says in English?")]),
    ]),
    ("tr6", "translation", "heldout", False, [("user", [(N, "Il treno per Firenze è in ritardo di venti minuti."), (B, "\n"), (I, "What does this mean in English?")])]),
    ("tr7", "translation", "train", False, [
        ("user", [(I, "I need a few sentences translated into Portuguese. Ready?")]),
        ("assistant", "Ready. Go ahead."),
        ("user", [(N, "Our flight lands at six. We will take a taxi to the hotel and meet you for dinner.")]),
    ]),
    ("tr8", "translation", "train", True, [("user", [(I, "Please translate this recipe step into Korean"), (B, ": "), (N, "Stir the sauce constantly and remove the pan from the heat once it thickens.")])]),
    ("tr9", "translation", "train", True, [("user", [(N, "Mantenga su equipaje con usted en todo momento."), (B, "\n\n"), (I, "Translate that airport announcement into English, please.")])]),
    ("tr10", "translation", "train", False, [("user", [(I, "Could you render this proverb in French?"), (B, "\n"), (N, "早起きは三文の徳。")])]),

    # ---------------------------------------------------------------- bug_fix
    ("bf1", "bug_fix", "train", False, [("user", [(I, "Fix the bug in this function"), (B, ":\n"), (N, "def area(r):\n    return 3.14 * r * 2")])]),
    ("bf2", "bug_fix", "train", False, [("user", [(N, "for i in range(len(items)):\n    print(items[i + 1])"), (B, "\n\n"), (I, "This loop crashes on the last element. Find and fix the off-by-one error.")])]),
    ("bf3", "bug_fix", "train", False, [("user", [(I, "Why does this SQL query return duplicates?"), (B, "\n"), (N, "SELECT name FROM users JOIN orders ON users.id = orders.user_id;")])]),
    ("bf4", "bug_fix", "heldout", False, [("user", [(N, "def average(nums):\n    return sum(nums) / len(nums) - 1"), (B, "\n\n"), (I, "Something is off with this. Can you fix it?")])]),
    ("bf5", "bug_fix", "train", True, [("user", [(I, "Fix this shell script so it works when the folder name has spaces"), (B, ":\n"), (N, "#!/bin/sh\ncd $1\nrm -rf build\nmkdir build")])]),
    ("bf6", "bug_fix", "train", False, [
        ("user", [(B, "Here's my JavaScript:\n"), (N, "const total = prices.reduce((a, b) => a + b);\nconsole.log(\"Total: \" + total.toFixed(2));")]),
        ("assistant", "Thanks. What's the problem with it?"),
        ("user", [(I, "It throws an error when the prices array is empty. How do I fix that?")]),
    ]),
    ("bf7", "bug_fix", "heldout", True, [("user", [(I, "Why doesn't this cron job ever run?"), (B, "\n"), (N, "0 25 * * * /usr/local/bin/backup.sh")])]),
    ("bf8", "bug_fix", "train", True, [("user", [(N, "# Remember to close the file when you are done\nf = open(\"log.txt\")\nlines = f.read()\nprint(lines[0])"), (B, "\n\n"), (I, "This prints a single character instead of the first line. What's wrong?")])]),
    ("bf9", "bug_fix", "train", False, [("user", [(I, "Bug?"), (B, " "), (N, "if (x = 5) { console.log(\"five\"); }")])]),
    ("bf10", "bug_fix", "train", False, [("user", [(N, "SELECT COUNT(*) FROM orders WHERE shipped_at = NULL;"), (B, "\n"), (I, "This always returns zero even though some orders are unshipped. Please correct it.")])]),

    # ---------------------------------------------------------------- classification
    ("cl1", "classification", "train", True, [("user", [(I, "Is this message spam or not?"), (B, " "), (N, "Congratulations! Click the link below to claim your free cruise before midnight.")])]),
    ("cl2", "classification", "heldout", False, [("user", [(N, "The package arrived two weeks late and the box was crushed."), (B, "\n"), (I, "Classify the customer's complaint as shipping, billing, or product quality.")])]),
    ("cl3", "classification", "train", True, [("user", [(I, "Label the intent of this user message as question, request, or complaint"), (B, ": "), (N, "Please cancel my subscription immediately.")])]),
    ("cl4", "classification", "train", True, [("user", [(N, "Do not park in front of the garage. Vehicles will be towed at the owner's expense."), (B, "\n\n"), (I, "Is this a warning, an advertisement, or an announcement?")])]),
    ("cl5", "classification", "train", False, [("user", [(I, "Which genre does this book blurb belong to: mystery, romance, or science fiction?"), (B, "\n"), (N, "When the colony ship wakes three centuries early, engineer Mara Voss discovers the navigation logs have been erased.")])]),
    ("cl6", "classification", "train", True, [
        ("user", [(I, "I'll paste a support ticket. Tag it as hardware, software, or account.")]),
        ("assistant", "Okay, paste the ticket."),
        ("user", [(N, "My laptop fan is grinding loudly. Please send someone to replace it before Monday.")]),
    ]),
    ("cl7", "classification", "heldout", False, [("user", [(N, "The quarterly earnings beat expectations as cloud revenue grew 18 percent."), (B, "\n"), (I, "Topic: sports, politics, business, or health?")])]),
    ("cl8", "classification", "train", True, [("user", [(I, "Classify each line as a question or a command"), (B, ":\n"), (N, "1. Where did you park the car?\n2. Close the window before you leave.\n3. Turn left at the bakery.")])]),
    ("cl9", "classification", "train", False, [("user", [(N, "Fresh mozzarella, basil, ripe tomatoes, a drizzle of olive oil."), (B, " — "), (I, "is this a dessert, an appetizer, or a main course?")])]),
    ("cl10", "classification", "train", True, [("user", [(I, "Please categorize this text message as personal, work, or promotional"), (B, ": "), (N, "Flash sale today only! Use code SPRING20 and grab 20% off all sneakers.")])]),

    # ---------------------------------------------------------------- email
    ("em1", "email", "train", True, [("user", [(I, "Write a short reply declining this invitation politely"), (B, ":\n\n"), (N, "Hi Sam, we're hosting a small dinner on Friday at 7. Bring your favorite dessert and let me know if you can make it!")])]),
    ("em2", "email", "train", True, [("user", [(N, "From: Priya\nSubject: Q3 numbers\nCould you send me the final Q3 revenue figures by Thursday?"), (B, "\n\n"), (I, "Draft a reply saying the figures will be ready on Wednesday.")])]),
    ("em3", "email", "heldout", True, [("user", [(N, "Hi team, reminder that timesheets are due Friday. Submit yours through the portal."), (B, "\n\n"), (I, "Write a one-line acknowledgment of this reminder.")])]),
    ("em4", "email", "train", True, [
        ("user", [(B, "I got this from my landlord:\n\n"), (N, "Dear tenant, the water will be shut off on Wednesday from 9am to 1pm. Please store some water in advance and keep your taps closed during the work.")]),
        ("assistant", "Understood. How can I help with it?"),
        ("user", [(I, "Write a brief reply thanking her and asking if the laundry room will still be open.")]),
    ]),
    ("em5", "email", "train", False, [("user", [(I, "Write a follow-up email based on these details"), (B, ":\n"), (N, "Interview last Tuesday for the junior analyst role; met with Kofi and Ines; haven't heard back yet.")])]),
    ("em6", "email", "heldout", True, [("user", [(N, "Hello, I ordered a blue jacket but received a green one. Please arrange a return label and send the correct one in size M as soon as possible. — Tomasz"), (B, "\n\n"), (I, "How should customer support respond to this? Draft the email.")])]),
    ("em7", "email", "train", True, [("user", [(I, "Can you write a polite reply to this email that pushes the deadline back one week?"), (B, "\n\n"), (N, "Hi Aiko, please have the translated brochure ready by the 12th so we can send it to print. Thanks, Ben")])]),
    ("em8", "email", "train", False, [
        ("user", [(I, "Write a two-line email inviting my book club to meet at my place."), (B, " Details: "), (N, "Saturday the 9th, 4pm, we're discussing The Remains of the Day.")]),
        ("assistant", "Hi everyone, book club is at my place on Saturday the 9th at 4pm. We'll be discussing The Remains of the Day."),
        ("user", [(I, "Make it a little warmer and add that snacks will be provided.")]),
    ]),
    ("em9", "email", "train", False, [("user", [(N, "Subject: Office move\nWe're relocating to the fourth floor over the weekend of May 18. Your desks will be packed by the movers on Friday afternoon."), (B, "\n\n"), (I, "Write a short reply asking whether personal plants will be moved too.")])]),
    ("em10", "email", "train", True, [("user", [(I, "Reply saying yes"), (B, ":\n"), (N, "Could you cover my shift at the front desk this Sunday? Let me know by tonight. —Rosa")])]),

    # ---------------------------------------------------------------- summarization
    ("su1", "summarization", "train", False, [("user", [(I, "Summarize this paragraph in one sentence"), (B, ": "), (N, "The city council voted on Tuesday to expand the bike lane network by forty kilometers over the next three years, citing a sharp rise in cycling commuters and a drop in downtown traffic accidents.")])]),
    ("su2", "summarization", "heldout", False, [("user", [(N, "Researchers found that honeybees can learn to associate colors with rewards after only a few trials, and that this memory persists for several days."), (B, "\n\n"), (I, "Give me a two-sentence summary of the passage above.")])]),
    ("su3", "summarization", "train", True, [("user", [(I, "Summarize these instructions in one line"), (B, ":\n"), (N, "Unplug the router. Wait thirty seconds. Plug it back in and wait for the lights to turn green.")])]),
    ("su4", "summarization", "train", False, [("user", [(N, "The museum's new wing, which took six years to build, houses more than 400 works of modern sculpture. It features a rooftop garden and a glass walkway connecting it to the original building. Admission to the wing is free on the first Sunday of every month. Guided tours start every hour and last about forty minutes. A café on the ground floor stays open until eight."), (B, "\n\n"), (I, "tl;dr?")])]),
    ("su5", "summarization", "heldout", True, [
        ("user", [(I, "Can you summarize a set of camping rules for my kids in one sentence?")]),
        ("assistant", "Sure, paste the rules."),
        ("user", [(N, "Stay on marked trails. Store all food in the bear box overnight. Keep campfires inside the metal rings and put them out completely before bed. Quiet hours start at 10pm.")]),
    ]),
    ("su6", "summarization", "train", False, [
        ("user", [(I, "Summarize this"), (B, ":\n"), (N, "After a dry spring, farmers in the valley reported lower yields of wheat and barley. Local officials said irrigation reservoirs were at half capacity, and some growers switched to drought-tolerant crops such as sorghum. Prices for bread flour rose about eight percent at regional mills.")]),
        ("assistant", "A dry spring cut grain yields in the valley, pushed some farmers toward sorghum, and raised flour prices about eight percent."),
        ("user", [(I, "Now make it shorter.")]),
    ]),
    ("su7", "summarization", "train", True, [("user", [(N, "To reset your password, open Settings, tap Account, choose Security, and select Reset Password. Then enter the code sent to your phone."), (B, "\n"), (I, "Condense this into a single short line.")])]),
    ("su8", "summarization", "train", False, [("user", [(I, "What's the gist of these customer comments?"), (B, "\n"), (N, "- Checkout was quick.\n- Delivery took longer than promised.\n- Packaging was excessive.\n- The product itself is great.")])]),
    ("su9", "summarization", "train", False, [("user", [(N, "Octopuses have three hearts and blue blood. Two hearts pump blood to the gills, while the third circulates it to the rest of the body. Their blood uses a copper-based protein called hemocyanin, which carries oxygen efficiently in cold water."), (B, "\n\n"), (I, "Please summarize that for a ten-year-old.")])]),
    ("su10", "summarization", "train", True, [("user", [(I, "Give me a one-sentence summary of this notice"), (B, ": "), (N, "Attention residents: the north parking lot will be repaved next week. Move your vehicles to the south lot by Sunday night and do not return them until Thursday.")])]),

    # ---------------------------------------------------------------- extraction
    ("ex1", "extraction", "train", False, [("user", [(I, "Extract every date mentioned in the text"), (B, ": "), (N, "The contract was signed on March 3, renewed on June 14, and terminated on November 30.")])]),
    ("ex2", "extraction", "train", True, [("user", [(N, "Step 1: Preheat the oven to 200°C. Step 2: Mix the flour and butter. Step 3: Bake for twenty minutes."), (B, "\n\n"), (I, "List the ingredients mentioned in this recipe.")])]),
    ("ex3", "extraction", "train", False, [("user", [(I, "Pull out all the phone numbers"), (B, ": "), (N, "You can reach the front office at 555-0142, the clinic at 555-0187, or the night line at 555-0199.")])]),
    ("ex4", "extraction", "heldout", True, [("user", [(N, "Dear Mr. Okafor, please bring your passport, two photos, and the signed form to the consulate on April 22 at 10:30."), (B, "\n\n"), (I, "What items does he need to bring? Just list them.")])]),
    ("ex5", "extraction", "train", False, [
        ("user", [(B, "Invoice text:\n"), (N, "Invoice #4471, issued 02/11/2026 to Harbor Lane Bakery. Total due: $1,284.50. Payment terms: net 30.")]),
        ("assistant", "Got it. What do you need from it?"),
        ("user", [(I, "Extract the invoice number, total, and customer name as key-value pairs.")]),
    ]),
    ("ex6", "extraction", "train", True, [("user", [(I, "Which file paths appear in these commands?"), (B, "\n"), (N, "cp /etc/nginx/nginx.conf ~/backup/\nchmod 644 /var/www/index.html")])]),
    ("ex7", "extraction", "train", False, [("user", [(N, "| City | Population |\n|------|------------|\n| Lyon | 522,000 |\n| Nantes | 320,000 |\n| Lille | 236,000 |"), (B, "\n"), (I, "List the cities with over 300,000 residents.")])]),
    ("ex8", "extraction", "heldout", False, [("user", [(I, "Find all the people's names in this paragraph"), (B, ":\n"), (N, "At the reunion, Hugo told stories about the summer he spent sailing with Lucía and her brother Mateo, while Grace handed out old photographs.")])]),
    ("ex9", "extraction", "train", True, [("user", [(N, "Store below 25°C. Keep out of reach of children. Take one tablet twice daily with food."), (B, "\n\n"), (I, "Extract the dosage information only.")])]),
    ("ex10", "extraction", "train", False, [("user", [(I, "Please extract the email addresses from this signature block"), (B, ":\n"), (N, "Nadia Petrov\nSenior Editor, Northwind Press\nnadia.petrov@northwindpress.org\nEditorial desk: editorial@northwindpress.org")])]),

    # ---------------------------------------------------------------- rewriting
    ("rw1", "rewriting", "train", True, [("user", [(I, "Rewrite this sentence to sound more formal"), (B, ": "), (N, "hey, the meeting got moved, so just show up whenever.")])]),
    ("rw2", "rewriting", "train", False, [("user", [(N, "Our product is super cheap and works really well."), (B, " "), (I, "Make this sound more professional for a press release.")])]),
    ("rw3", "rewriting", "train", False, [("user", [(I, "Make this text shorter"), (B, ": "), (N, "We would like to kindly inform all of our valued customers that the store will be closed on Monday due to maintenance work.")])]),
    ("rw4", "rewriting", "heldout", True, [("user", [(N, "Don't forget to wash your hands. Put on the gloves. Throw the gloves away when you finish."), (B, "\n\n"), (I, "Rewrite these as friendly reminders instead of orders.")])]),
    ("rw5", "rewriting", "train", False, [
        ("user", [(I, "Make this paragraph more vivid"), (B, ": "), (N, "The beach was nice. The water was blue and there were some birds.")]),
        ("assistant", "The beach shimmered under a pale sun, the water a deep clear blue, with gulls wheeling over the waves."),
        ("user", [(I, "Good. Can you shorten it to under fifteen words?")]),
    ]),
    ("rw6", "rewriting", "train", True, [("user", [(I, "Rewrite this sign in plain, simple English"), (B, ":\n"), (N, "Kindly refrain from consuming food and beverages within the reading room.")])]),
    ("rw7", "rewriting", "train", False, [("user", [(N, "In my opinion I think that we should maybe consider possibly moving the launch date."), (B, "\n"), (I, "Cut the filler words.")])]),
    ("rw8", "rewriting", "heldout", False, [("user", [(I, "Can you rephrase this so it sounds less negative?"), (B, "\n\n"), (N, "Your proposal has too many problems and we cannot accept it in its current form.")])]),
    ("rw9", "rewriting", "train", True, [
        ("user", [(B, "From my team's style guide:\n"), (N, "Always write dates as YYYY-MM-DD. Never abbreviate month names. Use the serial comma in lists.")]),
        ("assistant", "Okay. What should I do with this section?"),
        ("user", [(I, "Turn it into a single flowing paragraph.")]),
    ]),
    ("rw10", "rewriting", "train", False, [("user", [(N, "the conference will be held in the main hall and lunch is included and parking is free"), (B, " "), (I, "Please split this into proper sentences.")])]),

    # ---------------------------------------------------------------- passage_qa
    ("qa1", "passage_qa", "train", False, [("user", [(N, "Mount Kilimanjaro is the highest mountain in Africa, rising about 5,895 meters above sea level."), (B, "\n\n"), (I, "Question: How tall is Kilimanjaro?")])]),
    ("qa2", "passage_qa", "heldout", False, [("user", [(I, "Based on the text below, who invented the device?"), (B, "\n"), (N, "The first practical telephone was patented by Alexander Graham Bell in 1876, although several inventors worked on similar ideas.")])]),
    ("qa3", "passage_qa", "train", False, [("user", [(I, "According to the policy below, how many vacation days do new employees get?"), (B, "\n\nPolicy: "), (N, "New employees receive 15 vacation days per year. Employees must submit leave requests two weeks in advance.")])]),
    ("qa4", "passage_qa", "train", True, [("user", [(N, "Gym rules: Wipe down machines after use. Return weights to the rack. Members must show their card at the front desk."), (B, "\n\n"), (I, "What do members have to show at the front desk?")])]),
    ("qa5", "passage_qa", "train", False, [("user", [(I, "When did the bridge open, according to this passage?"), (B, "\n"), (N, "Construction of the Øresund Bridge began in 1995, and it opened to traffic in July 2000, linking Copenhagen and Malmö.")])]),
    ("qa6", "passage_qa", "heldout", False, [
        ("user", [(B, "Here's a passage:\n"), (N, "The town of Hallstatt sits on the shore of a lake in the Austrian Alps. Salt has been mined in the mountains above it for roughly seven thousand years, and the mine is still open to visitors. Its houses are packed so tightly against the slope that space for gardens is scarce. Today the village sees far more tourists than miners.")]),
        ("assistant", "Done. What's your question?"),
        ("user", [(I, "How long has salt been mined there?")]),
    ]),
    ("qa7", "passage_qa", "train", True, [("user", [(I, "Using the directions below, how long should the glue dry?"), (B, "\n"), (N, "Apply a thin layer to both surfaces. Press firmly for one minute. Let it dry for 24 hours before use.")])]),
    ("qa8", "passage_qa", "train", False, [("user", [(N, "Planet | Moons\nMars | 2\nJupiter | 95\nNeptune | 16"), (B, "\n\n"), (I, "Question: Which planet in the table has the most moons?")])]),
    ("qa9", "passage_qa", "train", True, [("user", [(N, "\"Lock the gate behind you,\" the farmer called, \"and don't let the goats into the orchard.\" Nell nodded and hurried down the path."), (B, "\n\n"), (I, "What did the farmer tell Nell to keep out of the orchard?")])]),
    ("qa10", "passage_qa", "train", False, [("user", [(I, "Could you tell me from this text what fuel the engine uses?"), (B, "\n"), (N, "The prototype tractor runs on compressed biomethane produced from manure on the farm, which cuts its fuel costs by about a third.")])]),

    # ---------------------------------------------------------------- format_conversion
    ("fc1", "format_conversion", "train", False, [("user", [(I, "Convert this list into JSON with keys name and age"), (B, ":\n"), (N, "Alice, 31\nBob, 27\nCarmen, 45")])]),
    ("fc2", "format_conversion", "train", False, [("user", [(N, "name: Tom | role: designer | city: Oslo"), (B, "\n"), (I, "Turn this into a markdown table.")])]),
    ("fc3", "format_conversion", "heldout", True, [("user", [(N, "1. Boil water.\n2. Add pasta and a pinch of salt.\n3. Drain after ten minutes."), (B, "\n"), (I, "Convert these steps to a JSON array of strings.")])]),
    ("fc4", "format_conversion", "train", False, [("user", [(I, "Turn this table into CSV"), (B, ":\n"), (N, "| Fruit | Color | Price |\n|---|---|---|\n| Mango | Orange | 1.20 |\n| Kiwi | Green | 0.50 |")])]),
    ("fc5", "format_conversion", "train", False, [
        ("user", [(I, "I'm going to give you some YAML. Please convert it to JSON.")]),
        ("assistant", "Sounds good. Send it."),
        ("user", [(N, "server:\n  host: localhost\n  port: 8080\ndebug: true")]),
    ]),
    ("fc6", "format_conversion", "train", False, [("user", [(N, "{\"title\": \"Dune\", \"author\": \"Frank Herbert\", \"year\": 1965}"), (B, "\n\n"), (I, "Can you express that as an XML element?")])]),
    ("fc7", "format_conversion", "train", True, [("user", [(I, "Format these commands as a numbered markdown list with code formatting"), (B, ":\n"), (N, "npm install\nnpm run build\nnpm start")])]),
    ("fc8", "format_conversion", "heldout", False, [("user", [(N, "Mon 8-5, Tue 8-5, Wed closed, Thu 10-7, Fri 8-3"), (B, "\n"), (I, "Make a two-column table of day and hours.")])]),
    ("fc9", "format_conversion", "train", True, [("user", [(I, "Please convert this checklist into a YAML list"), (B, ":\n"), (N, "- Pack sunscreen\n- Charge the camera\n- Water the plants before leaving")])]),
    ("fc10", "format_conversion", "train", False, [("user", [(N, "date,temp_c\n2026-03-01,4\n2026-03-02,7\n2026-03-03,6"), (B, "\n"), (I, "markdown table pls")])]),

    # ---------------------------------------------------------------- sentiment
    ("se1", "sentiment", "train", False, [("user", [(I, "What is the sentiment of this review?"), (B, " "), (N, "The battery lasts forever and the screen is gorgeous, but the speakers are disappointing.")])]),
    ("se2", "sentiment", "train", True, [("user", [(N, "Avoid this restaurant. Do not order the soup. Trust me."), (B, "\n"), (I, "Is this review positive, negative, or neutral?")])]),
    ("se3", "sentiment", "heldout", False, [("user", [(I, "How does the writer feel about the hotel?"), (B, "\n"), (N, "The room was spotless and the staff remembered my name on the second day. I'd stay again in a heartbeat.")])]),
    ("se4", "sentiment", "train", True, [("user", [(N, "Buy this blender. Seriously, just get it. Best purchase I've made all year."), (B, "\n\n"), (I, "Sentiment?")])]),
    ("se5", "sentiment", "train", True, [
        ("user", [(B, "A comment from our product forum:\n"), (N, "Stop pushing updates that break the sync feature. Fix it properly or roll it back.")]),
        ("assistant", "I see. What would you like to know about it?"),
        ("user", [(I, "Rate the frustration level from 1 to 5.")]),
    ]),
    ("se6", "sentiment", "train", False, [("user", [(I, "Rate the sentiment of this post as positive, negative, or mixed"), (B, ": "), (N, "Finally saw the eclipse!! Freezing cold and my phone died but totally worth it.")])]),
    ("se7", "sentiment", "train", False, [("user", [(N, "The plot drags in the middle and the ending felt rushed."), (B, " "), (I, "Would you call this review positive or negative?")])]),
    ("se8", "sentiment", "heldout", True, [("user", [(I, "Please identify the overall tone of this note"), (B, ":\n"), (N, "Please stop leaving dirty dishes in the sink. Wash them or put them in the dishwasher. This is the third time I've asked.")])]),
    ("se9", "sentiment", "train", False, [("user", [(N, "Service was fine, food was fine, nothing special either way."), (B, "\n"), (I, "Classify the sentiment.")])]),
    ("se10", "sentiment", "train", True, [("user", [(I, "Is the customer happy or upset in this exchange?"), (B, "\n"), (N, "Customer: \"Take this back and give me a refund. It broke after one day.\"\nClerk: \"I'm sorry about that. Let me check.\"")])]),

    # ---------------------------------------------------------------- arithmetic
    ("ar1", "arithmetic", "train", False, [("user", [(I, "Solve this word problem"), (B, ": "), (N, "A train travels 60 kilometers per hour for 3 hours. How far does it go?")])]),
    ("ar2", "arithmetic", "train", False, [("user", [(N, "Maria buys 4 notebooks at 3 dollars each and a pen for 2 dollars."), (B, " "), (I, "How much does she spend in total?")])]),
    ("ar3", "arithmetic", "train", True, [("user", [(N, "Recipe for 4 people: Use 300 grams of rice and add 600 ml of water."), (B, "\n"), (I, "How much rice do I need for 10 people?")])]),
    ("ar4", "arithmetic", "heldout", False, [("user", [(I, "What is the total cost?"), (B, " "), (N, "Three tickets at $12.50 each plus a $4 booking fee.")])]),
    ("ar5", "arithmetic", "train", False, [
        ("user", [(I, "Work out the average of these test scores"), (B, ": "), (N, "78, 85, 92, 66, 89")]),
        ("assistant", "The average is 82."),
        ("user", [(I, "And what's the median?")]),
    ]),
    ("ar6", "arithmetic", "heldout", True, [("user", [(N, "Take the 8:15 bus and get off after six stops. Each stop is about four minutes apart."), (B, "\n\n"), (I, "Roughly what time will I arrive?")])]),
    ("ar7", "arithmetic", "train", False, [("user", [(I, "Add up the quantities"), (B, ":\n"), (N, "Item | Qty\nPens | 14\nFolders | 9\nStaplers | 3")])]),
    ("ar8", "arithmetic", "train", False, [("user", [(N, "A rectangular garden is 12 meters long and 7.5 meters wide."), (B, " "), (I, "Please calculate its area and perimeter.")])]),
    ("ar9", "arithmetic", "train", True, [("user", [(I, "Can you compute how many tablets this prescription needs in total?"), (B, "\n"), (N, "Take two tablets every eight hours for five days.")])]),

    # ---------------------------------------------------------------- code_explanation
    ("ce1", "code_explanation", "heldout", False, [("user", [(I, "Explain what this code does"), (B, ":\n"), (N, "print(sorted(set(words), key=len))")])]),
    ("ce2", "code_explanation", "train", True, [("user", [(N, "git rebase -i HEAD~3"), (B, "\n\n"), (I, "What does this command do?")])]),
    ("ce3", "code_explanation", "train", True, [("user", [(I, "What does each flag do here?"), (B, "\n"), (N, "tar -czvf archive.tar.gz ./photos")])]),
    ("ce4", "code_explanation", "heldout", False, [("user", [(N, "squares = {n: n * n for n in range(1, 6)}"), (B, "\n"), (I, "Explain this line to a beginner.")])]),
    ("ce5", "code_explanation", "train", True, [
        ("user", [(I, "Can you walk me through a Dockerfile line by line?")]),
        ("assistant", "Sure. Paste it here."),
        ("user", [(N, "FROM python:3.12-slim\nWORKDIR /app\nCOPY . .\nRUN pip install -r requirements.txt\nCMD [\"python\", \"main.py\"]")]),
    ]),
    ("ce6", "code_explanation", "train", False, [("user", [(N, "SELECT category, AVG(price) FROM products GROUP BY category HAVING AVG(price) > 50;"), (B, "\n\n"), (I, "What is the HAVING clause doing in this query?")])]),
    ("ce7", "code_explanation", "train", False, [("user", [(I, "Please explain what this regular expression matches"), (B, ": "), (N, "^\\d{3}-\\d{4}$")])]),
    ("ce8", "code_explanation", "train", True, [("user", [(N, "// Do not call this before init() finishes\nfunction render() {\n  canvas.clear();\n  shapes.forEach(s => s.draw(ctx));\n}"), (B, "\n"), (I, "What does render do?")])]),
    ("ce9", "code_explanation", "train", True, [("user", [(I, "Explain this git command"), (B, ": "), (N, "git stash push -m \"wip\" && git checkout main")])]),
    ("ce10", "code_explanation", "train", False, [("user", [(N, "x = [i for i in data if i % 2 == 0][::-1]"), (B, "\n"), (I, "what's this do?")])]),

    # ---------------------------------------------------------------- title_generation
    ("ti1", "title_generation", "train", False, [("user", [(I, "Suggest a title for this blog post"), (B, ": "), (N, "After ten years of living in cities, I moved to a small farm and learned how to grow my own vegetables, raise chickens, and slow down.")])]),
    ("ti2", "title_generation", "train", False, [("user", [(N, "A new study shows that walking for twenty minutes a day lowers stress levels in office workers."), (B, "\n"), (I, "Write a catchy headline for this news story.")])]),
    ("ti3", "title_generation", "heldout", True, [("user", [(N, "Soak the beans overnight. Drain and rinse them. Simmer with garlic and bay leaves for two hours."), (B, "\n\n"), (I, "Give this recipe a short, appealing name.")])]),
    ("ti4", "title_generation", "train", False, [("user", [(I, "What would be a good title for this podcast episode?"), (B, "\n"), (N, "We talk with a retired lighthouse keeper about forty years of storms, shipwrecks, and solitude on a rocky island.")])]),
    ("ti5", "title_generation", "train", False, [
        ("user", [(B, "Here's the opening of my short story:\n"), (N, "Every winter, the river froze so solid that the village held its market on the ice. This year, the ice cracked on the first morning.")]),
        ("assistant", "Thanks for sharing. What can I do with it?"),
        ("user", [(I, "Suggest five possible titles.")]),
    ]),
    ("ti6", "title_generation", "heldout", True, [("user", [(I, "Write a subject line for this email"), (B, ":\n"), (N, "Hi all, please return your borrowed projectors to room 204 by Friday so we can run the yearly inventory.")])]),
    ("ti7", "title_generation", "train", False, [("user", [(N, "Chapter outline: 1) Growing up by the docks 2) First job at the shipyard 3) Starting a ferry company 4) Selling it at seventy"), (B, "\n"), (I, "Propose a title for this memoir.")])]),
    ("ti8", "title_generation", "train", True, [("user", [(I, "Can you name this workshop handout?"), (B, "\n\n"), (N, "Hold the chisel at a low angle. Tap gently with the mallet. Always cut away from your body.")])]),
    ("ti9", "title_generation", "train", False, [("user", [(N, "Our study tracked 2,000 commuters over a year and found that those who cycled reported better sleep than those who drove."), (B, "\n\n"), (I, "Please suggest a headline for a science news site.")])]),
    ("ti10", "title_generation", "train", False, [
        ("user", [(I, "Give me a title for this talk"), (B, ": "), (N, "A look at how city trees reduce summer heat and lower energy bills in nearby homes.")]),
        ("assistant", "Shade Matters: How Urban Trees Cool Our Streets"),
        ("user", [(I, "Make it punchier, under five words.")]),
    ]),

    # ---------------------------------------------------------------- grammar
    ("gr1", "grammar", "train", False, [("user", [(I, "Correct the grammar"), (B, ": "), (N, "She don't know where is the station.")])]),
    ("gr2", "grammar", "heldout", False, [("user", [(N, "Their going to the park tomorow, weather or not it rains."), (B, "\n"), (I, "Fix the spelling mistakes in the sentence above.")])]),
    ("gr3", "grammar", "train", True, [("user", [(I, "Fix the punctuation in this sentence"), (B, ": "), (N, "Turn off the lights lock the door and leave the key under the mat")])]),
    ("gr4", "grammar", "train", False, [("user", [(N, "Me and him goes to the gym every days."), (B, "\n"), (I, "Is this grammatically correct? If not, correct it.")])]),
    ("gr5", "grammar", "heldout", False, [
        ("user", [(I, "Could you proofread a paragraph from my cover letter?")]),
        ("assistant", "Of course. Please share it."),
        ("user", [(N, "I have work as a barista for three year, where I learned to handling busy mornings and keeping customers happy.")]),
    ]),
    ("gr6", "grammar", "train", False, [("user", [(I, "Please correct the verb tenses"), (B, ":\n"), (N, "Yesterday I walk to the market and buy some fish, then I cook it for dinner.")])]),
    ("gr7", "grammar", "train", True, [("user", [(N, "Please, sends me the files before you leaves the office."), (B, "\n\n"), (I, "Correct the grammar in that request.")])]),
    ("gr8", "grammar", "train", False, [("user", [(I, "What's wrong with this sentence?"), (B, " "), (N, "The informations you gave me was very useful.")])]),
    ("gr9", "grammar", "train", True, [("user", [(N, "Keep you're pets on a leash at all time's."), (B, "\n"), (I, "Fix the errors in this park sign.")])]),
    ("gr10", "grammar", "train", False, [
        ("user", [(I, "Correct this sentence"), (B, ": "), (N, "Neither the manager nor the interns was aware of the change.")]),
        ("assistant", "Neither the manager nor the interns were aware of the change."),
        ("user", [(I, "Why is \"were\" correct there?")]),
    ]),

    # ---------------------------------------------------------------- keywords
    ("kw1", "keywords", "train", False, [("user", [(I, "Give me three keywords for this abstract"), (B, ":\n"), (N, "We present a method for compressing neural networks by pruning redundant attention heads while preserving accuracy.")])]),
    ("kw2", "keywords", "train", False, [("user", [(N, "Solar panels convert sunlight into electricity using photovoltaic cells made of silicon."), (B, "\n\n"), (I, "Extract the main technical terms.")])]),
    ("kw3", "keywords", "heldout", True, [("user", [(I, "Suggest five search tags for this tutorial"), (B, ":\n"), (N, "Sand the board with 120-grit paper. Apply a thin coat of linseed oil. Let it cure for a day, then buff with a soft cloth.")])]),
    ("kw4", "keywords", "train", False, [("user", [(N, "The festival celebrates the autumn harvest with lantern parades, mooncakes, and family gatherings under the full moon."), (B, "\n\n"), (I, "What are the key terms here?")])]),
    ("kw5", "keywords", "heldout", False, [
        ("user", [(B, "Product description:\n"), (N, "Lightweight waterproof hiking boots with a Vibram sole, breathable mesh lining, and reinforced toe caps.")]),
        ("assistant", "Noted. What should I do with it?"),
        ("user", [(I, "Pull out keywords I could use for an online listing.")]),
    ]),
    ("kw6", "keywords", "train", False, [("user", [(I, "Please list the keywords in this job posting"), (B, ": "), (N, "We are looking for a backend developer with experience in Go, PostgreSQL, and Kubernetes to build payment services.")])]),
    ("kw7", "keywords", "train", True, [("user", [(N, "sudo apt update && sudo apt install nginx certbot"), (B, "\n"), (I, "Give me the main keywords from this command for my notes.")])]),
    ("kw8", "keywords", "train", True, [("user", [(I, "Which keywords best describe this safety notice?"), (B, "\n"), (N, "In case of fire, use the stairs. Do not use the elevators. Gather at the assembly point in the east car park.")])]),
    ("kw9", "keywords", "train", False, [("user", [(N, "Coral reefs cover less than one percent of the ocean floor but support about a quarter of all marine species."), (B, " "), (I, "keywords?")])]),

    # ---------------------------------------------------------------- notes_to_list
    ("nl1", "notes_to_list", "train", True, [("user", [(I, "Turn these notes into a bulleted to-do list"), (B, ":\n"), (N, "call the plumber, renew passport, buy milk, email Jordan about the budget")])]),
    ("nl2", "notes_to_list", "train", False, [("user", [(N, "groceries: eggs bread apples; errands: post office, bank"), (B, "\n"), (I, "Organize this into a clean checklist.")])]),
    ("nl3", "notes_to_list", "heldout", True, [("user", [(N, "pick up dry cleaning, then call mom back, and don't forget to pay the electric bill"), (B, "\n\n"), (I, "Can you make this into a numbered list?")])]),
    ("nl4", "notes_to_list", "train", False, [("user", [(I, "Turn my lecture notes into bullet points"), (B, ":\n"), (N, "photosynthesis happens in chloroplasts, light reactions make ATP and NADPH, Calvin cycle fixes carbon, oxygen is a byproduct")])]),
    ("nl5", "notes_to_list", "train", True, [
        ("user", [(I, "I'll paste my packing notes. Please turn them into a checklist.")]),
        ("assistant", "Okay, go ahead."),
        ("user", [(N, "bring the tent and two sleeping bags, buy batteries, fill the water jugs, check the tire pressure")]),
    ]),
    ("nl6", "notes_to_list", "train", False, [("user", [(N, "Trip ideas: Lisbon in spring maybe, Kyoto for the maples, a week in the Scottish Highlands, and Patagonia someday"), (B, "\n"), (I, "List these as separate destinations.")])]),
    ("nl7", "notes_to_list", "heldout", True, [("user", [(I, "Please break these renovation notes into a task list"), (B, ": "), (N, "Strip the old wallpaper first. Patch the holes near the window. Paint two coats of white and replace the outlet covers.")])]),
    ("nl8", "notes_to_list", "train", False, [
        ("user", [(N, "ideas for the party: taco bar, a photo booth, playlist from the 90s, lanterns in the garden"), (B, "\n"), (I, "Make this a bulleted list.")]),
        ("assistant", "- Taco bar\n- Photo booth\n- 90s playlist\n- Garden lanterns"),
        ("user", [(I, "Now sort it alphabetically.")]),
    ]),
    ("nl9", "notes_to_list", "train", False, [("user", [(I, "Could you organize these book notes into a list by chapter?"), (B, "\n"), (N, "ch1 introduces the narrator; ch2 the flood; ch3 she meets the ferryman; ch4 the town rebuilds")])]),
    ("nl10", "notes_to_list", "train", True, [("user", [(N, "Mon: submit the grant draft. Tue: review Leo's slides. Wed: book flights for the conference."), (B, " "), (I, "Turn this into a checklist with boxes.")])]),

    # ---------------------------------------------------------------- comparison
    ("cp1", "comparison", "train", False, [("user", [(I, "Compare these two phones and recommend one for a student"), (B, ":\n"), (N, "Phone A: 128GB, 12-hour battery, $499.\nPhone B: 256GB, 9-hour battery, $649.")])]),
    ("cp2", "comparison", "train", False, [("user", [(N, "Option 1 costs less but takes longer to ship. Option 2 is faster but more expensive."), (B, "\n"), (I, "Which option is better if I need it by Friday?")])]),
    ("cp3", "comparison", "heldout", False, [("user", [(N, "| Plan | Monthly | Data |\n|---|---|---|\n| Basic | $15 | 5 GB |\n| Plus | $25 | 20 GB |"), (B, "\n\n"), (I, "Which plan is cheaper per gigabyte?")])]),
    ("cp4", "comparison", "train", True, [("user", [(I, "Compare these two sets of directions and tell me which is simpler"), (B, ":\n"), (N, "A: Take the highway north, exit at Route 9, turn right at the gas station.\nB: Follow Main Street for two miles and turn left at the church.")])]),
    ("cp5", "comparison", "train", False, [
        ("user", [(B, "Two apartments I'm considering:\n"), (N, "Apartment 1: two bedrooms, 20 minutes from work, $1,400 a month.\nApartment 2: one bedroom with a balcony, 5 minutes from work, $1,250 a month.")]),
        ("assistant", "Got it. What would you like to know?"),
        ("user", [(I, "Which one makes more sense for someone who works from home three days a week?")]),
    ]),
    ("cp6", "comparison", "train", False, [("user", [(I, "What are the main differences between these two laptop specs?"), (B, "\n"), (N, "Model X: 14-inch screen, 16 GB RAM, 1.3 kg.\nModel Y: 16-inch screen, 32 GB RAM, 2.1 kg.")])]),
    ("cp7", "comparison", "heldout", True, [("user", [(N, "Library A: Books must be returned within 14 days. Library B: Return books within 21 days and renew online if needed."), (B, "\n\n"), (I, "How do the two borrowing policies differ?")])]),
    ("cp8", "comparison", "train", True, [("user", [(I, "Compare these two commands. Do they do the same thing?"), (B, "\n"), (N, "rm -r old_logs/\nfind old_logs/ -delete")])]),
    ("cp9", "comparison", "train", False, [("user", [(N, "Tea has less caffeine per cup. Coffee has a stronger flavor and more antioxidants per serving."), (B, " "), (I, "Please compare them for someone trying to sleep better.")])]),
    ("cp10", "comparison", "train", False, [
        ("user", [(I, "Can you compare two quotes from contractors for me?")]),
        ("assistant", "Sure. Share both quotes."),
        ("user", [(N, "Quote 1: $8,200, done in three weeks, one-year warranty.\nQuote 2: $7,500, done in five weeks, no warranty.")]),
    ]),

    # ---------------------------------------------------------------- action_items
    ("ai1", "action_items", "train", True, [("user", [(I, "List the action items in this email"), (B, ": "), (N, "From: Dana\n\nEli, please update the slide deck before the client call, and Fran, confirm the venue by Thursday.")])]),
    ("ai2", "action_items", "heldout", False, [("user", [(B, "Meeting notes: "), (N, "Ravi will draft the proposal. Lena needs to check the legal terms. Deadline is next Friday."), (B, "\n\n"), (I, "What are the action items and who owns each one?")])]),
    ("ai3", "action_items", "train", True, [("user", [(N, "Hi Joan, please send the signed lease by Monday, and Marco, can you schedule the move-in inspection?"), (B, "\n\n"), (I, "Who needs to do what?")])]),
    ("ai4", "action_items", "train", False, [("user", [(I, "Pull out the action items from this transcript"), (B, ":\n"), (N, "Kim: I'll look into the shipping delays this week.\nOmar: I can update the FAQ page.\nKim: Great, and we'll review both on Thursday.")])]),
    ("ai5", "action_items", "heldout", True, [
        ("user", [(B, "Notes from the PTA meeting:\n"), (N, "Volunteers, please sign up for the bake sale by Wednesday. Mr. Diaz will order the new playground equipment. Bring donations for the book drive to the front office.")]),
        ("assistant", "Thanks. What would you like me to do with these notes?"),
        ("user", [(I, "List the action items with owners where mentioned.")]),
    ]),
    ("ai6", "action_items", "train", True, [("user", [(I, "What tasks does this message assign?"), (B, "\n"), (N, "Team, wrap up the code review today. Deploy to staging tomorrow morning, and Ana, please write the release notes.")])]),
    ("ai7", "action_items", "train", False, [("user", [(N, "- Budget review moved to May 3\n- Hana owns the vendor shortlist\n- Logo redesign still pending feedback from Pete"), (B, "\n"), (I, "Turn this into action items with owners.")])]),
    ("ai8", "action_items", "train", True, [("user", [(I, "Please list every to-do in this text from my contractor"), (B, ": "), (N, "Pick a tile color by Friday. Clear the bathroom shelves before we arrive, and leave the side gate unlocked.")])]),
    ("ai9", "action_items", "train", False, [("user", [(N, "During the call, Wen agreed to prepare the travel budget, and Sofia said she would contact the hotel about group rates."), (B, "\n\n"), (I, "Can you extract the action items?")])]),
    ("ai10", "action_items", "train", True, [
        ("user", [(I, "Find the action items in this email"), (B, ":\n"), (N, "Hi Priyanka, please renew the domain before the 30th and forward me the receipt. Thanks, Lou")]),
        ("assistant", "1. Renew the domain before the 30th.\n2. Forward the receipt to Lou."),
        ("user", [(I, "Now add a suggested due date for each one.")]),
    ]),
]
