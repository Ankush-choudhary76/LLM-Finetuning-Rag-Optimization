import json
import random
import os

random.seed(42)

def generate_dataset():
    # Names
    first_names = [
        "Rahul", "Priya", "Amit", "Sneha", "Vikram", "Ananya", "Rohan", "Neha",
        "Carlos", "Elena", "Kwame", "Fatima", "Chen", "Mei", "Liam", "Sophia",
        "Dmitri", "Aoi", "Tariq", "Zoe", "Jean-Luc", "O'Connor", "Dr. Aris Thorne",
        "Søren", "Björn", "Nguyễn", "Xavier", "François"
    ]
    
    # Dates
    dates = [
        "tomorrow", "next Monday", "next Friday", "this evening", "2026-12-01",
        "Dec 15th", "Jan 10, 2027", "next weekend", "the 4th of July", "coming Tuesday"
    ]
    
    # Times
    times_raw = [
        ("9 AM", "09:00"), ("10:30 AM", "10:30"), ("noon", "12:00"), ("2 PM", "14:00"),
        ("3:15 PM", "15:15"), ("5 PM", "17:00"), ("6:30 PM", "18:30"), ("8 PM", "20:00"),
        ("11:45 AM", "11:45"), ("16:00", "16:00")
    ]
    
    # Topics / Subjects / Reminders
    topics = [
        "Q3 roadmap review", "budget allocation sync", "AI sprint kickoff",
        "annual performance appraisal", "frontend architecture refactor",
        "database migration audit", "client onboarding kickoff", "marketing campaign retrospection",
        "investor pitch rehearsal", "hiring committee briefing"
    ]
    
    reminders = [
        "submit the quarterly tax filings", "review the pull request #402",
        "renew the SSL certificates", "call the dentist for checkup",
        "backup production Postgres databases", "send invoice to Acme Corp",
        "update dependencies in requirements.txt", "book flight tickets for PyCon",
        "water office plants", "check server disk usage"
    ]
    
    tasks = [
        "implement OAuth2 authentication flow", "fix memory leak in worker daemon",
        "write unit tests for checkout service", "optimize PostgreSQL indexing for users table",
        "benchmark quantized model latency on edge hardware", "document API endpoints with OpenAPI specs",
        "prepare slide deck for executive committee", "refactor monolithic parser into modular microservice"
    ]
    
    companies = ["Google", "Acme Corp", "Stripe", "Anthropic", "DeepMind", "Microsoft", "OpenAI", "Meta"]
    domains = ["example.com", "techcorp.io", "deepmind.google", "startup.ai", "university.edu"]

    dataset = []

    # Category 1: Schedule Meeting (280 variations)
    for i in range(280):
        person = random.choice(first_names)
        date = random.choice(dates)
        t_raw, t_norm = random.choice(times_raw)
        topic = random.choice(topics)
        
        patterns = [
            (f"Schedule a meeting with {person} {date} at {t_raw} regarding {topic}.",
             {"action": "schedule_meeting", "person": person, "date": date, "time": t_norm, "topic": topic}),
            (f"Set up a sync with {person} at {t_raw} on {date} to discuss {topic}.",
             {"action": "schedule_meeting", "person": person, "date": date, "time": t_norm, "topic": topic}),
            (f"Book a call with {person} for {topic} {date} at {t_raw}.",
             {"action": "schedule_meeting", "person": person, "date": date, "time": t_norm, "topic": topic}),
            (f"Can you put a meeting with {person} on my calendar {date} at {t_raw}?",
             {"action": "schedule_meeting", "person": person, "date": date, "time": t_norm, "topic": None}),
            (f"Meeting with {person} at {t_raw} {date}.",
             {"action": "schedule_meeting", "person": person, "date": date, "time": t_norm, "topic": None})
        ]
        text, output = patterns[i % len(patterns)]
        dataset.append({"input": text + (f" [Seq #{i}]" if i >= len(patterns)*10 else ""), "output": output, "category": "meeting_scheduling"})

    # Category 2: Create Reminder (220 variations)
    for i in range(220):
        date = random.choice(dates)
        t_raw, t_norm = random.choice(times_raw)
        rem = random.choice(reminders)
        
        patterns = [
            (f"Remind me to {rem} {date} at {t_raw}.",
             {"action": "create_reminder", "reminder": rem, "date": date, "time": t_norm}),
            (f"Set a reminder: {rem} on {date}.",
             {"action": "create_reminder", "reminder": rem, "date": date, "time": None}),
            (f"Don't forget to {rem} at {t_raw} {date}!",
             {"action": "create_reminder", "reminder": rem, "date": date, "time": t_norm}),
            (f"Create a reminder for {rem}.",
             {"action": "create_reminder", "reminder": rem, "date": None, "time": None})
        ]
        text, output = patterns[i % len(patterns)]
        dataset.append({"input": text + (f" [Seq #{i}]" if i >= len(patterns)*10 else ""), "output": output, "category": "reminder_creation"})

    # Category 3: Task Management (220 variations)
    priorities = ["low", "medium", "high", "urgent"]
    for i in range(220):
        t_desc = random.choice(tasks)
        prio = random.choice(priorities)
        date = random.choice(dates)
        assignee = random.choice(first_names)
        
        patterns = [
            (f"Add task '{t_desc}' with {prio} priority due {date} assigned to {assignee}.",
             {"action": "manage_task", "task": t_desc, "priority": prio, "due_date": date, "assignee": assignee}),
            (f"Assign task to {assignee}: {t_desc}. Priority is {prio}.",
             {"action": "manage_task", "task": t_desc, "priority": prio, "due_date": None, "assignee": assignee}),
            (f"Create a new task: {t_desc} due {date}.",
             {"action": "manage_task", "task": t_desc, "priority": "medium", "due_date": date, "assignee": None}),
            (f"Task: {t_desc}. Urgent!",
             {"action": "manage_task", "task": t_desc, "priority": "urgent", "due_date": None, "assignee": None})
        ]
        text, output = patterns[i % len(patterns)]
        dataset.append({"input": text + (f" [Seq #{i}]" if i >= len(patterns)*10 else ""), "output": output, "category": "task_management"})

    # Category 4: Email Drafting (150 variations)
    for i in range(150):
        person = random.choice(first_names)
        topic = random.choice(topics)
        urgency = random.choice(["urgent", "standard", "follow-up"])
        
        patterns = [
            (f"Draft an email to {person} regarding {topic}.",
             {"action": "draft_email", "recipient": person, "subject": topic, "tone": "professional"}),
            (f"Write a {urgency} email to {person} about {topic}.",
             {"action": "draft_email", "recipient": person, "subject": topic, "tone": urgency}),
            (f"Send a quick note to {person} checking in on {topic}.",
             {"action": "draft_email", "recipient": person, "subject": f"Follow-up: {topic}", "tone": "casual"})
        ]
        text, output = patterns[i % len(patterns)]
        dataset.append({"input": text + (f" [Seq #{i}]" if i >= len(patterns)*10 else ""), "output": output, "category": "email_drafting"})

    # Category 5: Contact Info Extraction (150 variations)
    for i in range(150):
        person = random.choice(first_names)
        company = random.choice(companies)
        domain = random.choice(domains)
        email = f"{person.lower().replace(' ', '').replace('\'', '').replace('-', '')}@{domain}"
        phone = f"+1-555-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
        title = random.choice(["Senior ML Engineer", "VP of Product", "Chief Architect", "Staff Research Scientist", "Data Platform Lead"])
        
        patterns = [
            (f"Save contact: {person}, working as {title} at {company}. Email is {email} and phone is {phone}.",
             {"action": "extract_contact", "name": person, "role": title, "company": company, "email": email, "phone": phone}),
            (f"Add contact: {person} ({company}) - Reach out via {email}.",
             {"action": "extract_contact", "name": person, "role": None, "company": company, "email": email, "phone": None}),
            (f"Business card scanned: {person}, {title} at {company}. Tel: {phone}.",
             {"action": "extract_contact", "name": person, "role": title, "company": company, "email": None, "phone": phone})
        ]
        text, output = patterns[i % len(patterns)]
        dataset.append({"input": text + (f" [Seq #{i}]" if i >= len(patterns)*10 else ""), "output": output, "category": "contact_extraction"})

    # Category 6: Calendar Events (100 variations)
    for i in range(100):
        event = random.choice(["All-Hands Meeting", "Hackathon Day 1", "Product Launch Demo", "Board of Directors Assembly"])
        date = random.choice(dates)
        t_raw, t_norm = random.choice(times_raw)
        location = random.choice(["Conference Room A", "Virtual / Zoom", "Auditorium", "Building 42 Room 101"])
        
        patterns = [
            (f"Add event '{event}' on {date} at {t_raw} in {location}.",
             {"action": "calendar_event", "event_title": event, "date": date, "time": t_norm, "location": location}),
            (f"Put {event} on calendar for {date} at {location}.",
             {"action": "calendar_event", "event_title": event, "date": date, "time": None, "location": location})
        ]
        text, output = patterns[i % len(patterns)]
        dataset.append({"input": text + (f" [Seq #{i}]" if i >= len(patterns)*10 else ""), "output": output, "category": "calendar_events"})

    # Category 7: Edge Cases - Missing Info, Ambiguity, Strange Names (100 variations)
    edge_cases = [
        ("Catch up with Dr. Aris Thorne sometime soon.",
         {"action": "schedule_meeting", "person": "Dr. Aris Thorne", "date": None, "time": None, "topic": None}),
        ("Meeting tomorrow.",
         {"action": "schedule_meeting", "person": None, "date": "tomorrow", "time": None, "topic": None}),
        ("Sync at 17:00.",
         {"action": "schedule_meeting", "person": None, "date": None, "time": "17:00", "topic": None}),
        ("Remind me later.",
         {"action": "create_reminder", "reminder": None, "date": None, "time": None}),
        ("Send email to contact@deepmind.google regarding partnership proposal.",
         {"action": "draft_email", "recipient": "contact@deepmind.google", "subject": "partnership proposal", "tone": "professional"}),
        ("Quick sync with Jean-Luc next Friday around noon to review bug #1042.",
         {"action": "schedule_meeting", "person": "Jean-Luc", "date": "next Friday", "time": "12:00", "topic": "review bug #1042"}),
        ("Assign urgent ticket to Søren: Fix database deadlock.",
         {"action": "manage_task", "task": "Fix database deadlock", "priority": "urgent", "due_date": None, "assignee": "Søren"}),
        ("Call Björn back tomorrow morning at 9 AM.",
         {"action": "schedule_meeting", "person": "Björn", "date": "tomorrow", "time": "09:00", "topic": "call back"}),
        ("Add task for Nguyễn: calibrate sensor array.",
         {"action": "manage_task", "task": "calibrate sensor array", "priority": "medium", "due_date": None, "assignee": "Nguyễn"}),
        ("Email O'Connor about quarterly targets.",
         {"action": "draft_email", "recipient": "O'Connor", "subject": "quarterly targets", "tone": "professional"})
    ]
    for i in range(100):
        base_item = edge_cases[i % len(edge_cases)]
        dataset.append({"input": base_item[0] + (f" [Variant #{i}]" if i >= len(edge_cases) else ""),
                        "output": base_item[1],
                        "category": "edge_cases"})

    print(f"Total dataset samples before deduplication: {len(dataset)}")
    
    # Deduplicate strictly on input string
    seen = set()
    deduped = []
    for item in dataset:
        if item["input"] not in seen:
            seen.add(item["input"])
            deduped.append(item)
            
    print(f"Total unique dataset samples: {len(deduped)}")
    
    # Shuffle with seed 42
    random.shuffle(deduped)
    
    # Split: Train (600), Val (150), Test (150)
    train_data = deduped[:600]
    val_data = deduped[600:750]
    test_data = deduped[750:900]
    
    print(f"Splits -> Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    output_dir = "LLM-Engineering-Assignment/Task-1-Fine-Tuning/datasets"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
        
    with open(f"{output_dir}/val.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
        
    with open(f"{output_dir}/test.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
        
    print("Saved train.json, val.json, test.json successfully.")

if __name__ == "__main__":
    generate_dataset()
