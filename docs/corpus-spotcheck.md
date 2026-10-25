# Corpus v1 spot-check (20 random examples, seed 7)

Please skim these and flag anything that looks mislabeled. Key: **[I: …]** is the user's instruction, **[N: …]** is content the instruction operates on (even when it's phrased as a command), and **[B: …]** is a connector or framing. Assistant turns are shown in italics. The rule is *role, not mood*.

### ar2 · arithmetic · train
> **[N:** Maria buys 4 notebooks at 3 dollars each and a pen for 2 dollars.**]****[B:**  **]****[I:** How much does she spend in total?**]**

### ar8 · arithmetic · train
> **[N:** A rectangular garden is 12 meters long and 7.5 meters wide.**]****[B:**  **]****[I:** Please calculate its area and perimeter.**]**

### bf3 · bug_fix · train
> **[I:** Why does this SQL query return duplicates?**]****[B:**  ⏎ **]****[N:** SELECT name FROM users JOIN orders ON users.id = orders.user_id;**]**

### bf5 · bug_fix · train · imperative content
> **[I:** Fix this shell script so it works when the folder name has spaces**]****[B:** : ⏎ **]****[N:** #!/bin/sh ⏎ cd $1 ⏎ rm -rf build ⏎ mkdir build**]**

### bf8 · bug_fix · train · imperative content
> **[N:** # Remember to close the file when you are done ⏎ f = open("log.txt") ⏎ lines = f.read() ⏎ print(lines[0])**]****[B:**  ⏎  ⏎ **]****[I:** This prints a single character instead of the first line. What's wrong?**]**

### bf9 · bug_fix · train
> **[I:** Bug?**]****[B:**  **]****[N:** if (x = 5) { console.log("five"); }**]**

### ce3 · code_explanation · train · imperative content
> **[I:** What does each flag do here?**]****[B:**  ⏎ **]****[N:** tar -czvf archive.tar.gz ./photos**]**

### cl3 · classification · train · imperative content
> **[I:** Label the intent of this user message as question, request, or complaint**]****[B:** : **]****[N:** Please cancel my subscription immediately.**]**

### cl4 · classification · train · imperative content
> **[N:** Do not park in front of the garage. Vehicles will be towed at the owner's expense.**]****[B:**  ⏎  ⏎ **]****[I:** Is this a warning, an advertisement, or an announcement?**]**

### cl5 · classification · train
> **[I:** Which genre does this book blurb belong to: mystery, romance, or science fiction?**]****[B:**  ⏎ **]****[N:** When the colony ship wakes three centuries early, engineer Mara Voss discovers the navigation logs have been erased.**]**

### cp9 · comparison · train
> **[N:** Tea has less caffeine per cup. Coffee has a stronger flavor and more antioxidants per serving.**]****[B:**  **]****[I:** Please compare them for someone trying to sleep better.**]**

### em9 · email · train
> **[N:** Subject: Office move ⏎ We're relocating to the fourth floor over the weekend of May 18. Your desks will be packed by the movers on Friday afternoon.**]****[B:**  ⏎  ⏎ **]****[I:** Write a short reply asking whether personal plants will be moved too.**]**

### ex5 · extraction · train
> **[B:** Invoice text: ⏎ **]****[N:** Invoice #4471, issued 02/11/2026 to Harbor Lane Bakery. Total due: $1,284.50. Payment terms: net 30.**]**
> *assistant: Got it. What do you need from it?*
> **[I:** Extract the invoice number, total, and customer name as key-value pairs.**]**

### fc3 · format_conversion · heldout · imperative content
> **[N:** 1. Boil water. ⏎ 2. Add pasta and a pinch of salt. ⏎ 3. Drain after ten minutes.**]****[B:**  ⏎ **]****[I:** Convert these steps to a JSON array of strings.**]**

### gr1 · grammar · train
> **[I:** Correct the grammar**]****[B:** : **]****[N:** She don't know where is the station.**]**

### gr9 · grammar · train · imperative content
> **[N:** Keep you're pets on a leash at all time's.**]****[B:**  ⏎ **]****[I:** Fix the errors in this park sign.**]**

### nl2 · notes_to_list · train
> **[N:** groceries: eggs bread apples; errands: post office, bank**]****[B:**  ⏎ **]****[I:** Organize this into a clean checklist.**]**

### rw2 · rewriting · train
> **[N:** Our product is super cheap and works really well.**]****[B:**  **]****[I:** Make this sound more professional for a press release.**]**

### se4 · sentiment · train · imperative content
> **[N:** Buy this blender. Seriously, just get it. Best purchase I've made all year.**]****[B:**  ⏎  ⏎ **]****[I:** Sentiment?**]**

### tr10 · translation · train
> **[I:** Could you render this proverb in French?**]****[B:**  ⏎ **]****[N:** 早起きは三文の徳。**]**
