# SpecHO Documentation Map

Quick reference for which document to read based on what you need.

---

## Starting Development

**I want to start coding RIGHT NOW**
→ Read: `docs/QUICKSTART.md`

This gives you setup commands, Task 1.1 specification, and tells you exactly what to build first.

---

**I want to understand the project structure**
→ Read: `CLAUDE.md`

This explains the overall architecture, tier system, why three tiers exist, and where to find specific information.

---

**I want to see all 32 tasks**
→ Read: `docs/TASKS.md`

This is a complete task list in YAML format, showing dependencies between tasks and what each delivers.

---

## During Implementation

**I'm implementing Task X and need detailed specs**
→ Read: `SPECS.md`

Find your component's specifications by tier. See exactly what Tier 1, Tier 2, and Tier 3 look like for that component.

**I need to understand configuration options**
→ Read: `SPECS.md` (Configuration Profiles section)

Shows all three profiles (simple, robust, research) with their parameters and when to use each.

**I need to understand what tests to write**
→ Read: `SPECS.md` (Testing Strategy section)

Shows testing requirements by tier and what types of tests to write.

---

## After Tier 1 is Complete

**I want to deploy this to the web**
→ Read: `DEPLOYMENT.md` (Phase W1)

Gives you FastAPI setup, SQLite database structure, and simple HTML UI.

**I want to package this in Docker**
→ Read: `DEPLOYMENT.md` (Phase W2)

Provides Dockerfile and docker-compose.yml ready to use.

**I want to make this a pip-installable library**
→ Read: `DEPLOYMENT.md` (Alternative: Library Installation section)

Shows setup.py and how users would install and use SpecHO.

---

## Understanding Design Decisions

**Why did we choose three tiers?**
→ Read: `CLAUDE.md` (Philosophy section)

Explains the three-tier approach and why we build MVP first, then Production, then Research.

**How do I know when to move from Tier 1 to Tier 2?**
→ Read: `SPECS.md` (Tier Transition Metrics section)

Specific, measurable criteria for advancing between tiers.

**What's the philosophy of this project?**
→ Read: The architecture document in your project root (provided separately)

This explains the Echo Rule watermark detection algorithm itself.

---

## File Reference

```
Project Root/
├── CLAUDE.md                ← Architecture overview & navigation (READ THIS FIRST)
├── README.md                ← GitHub repository introduction
├── architecture.md          ← Original Echo Rule design document
│
├── docs/
│   ├── QUICKSTART.md        ← Implementation startup guide (5 min read)
│   └── TASKS.md             ← All 32 tasks in YAML format (reference)
│
├── SPECS.md                 ← Detailed tier specs by component (lookup as needed)
├── DEPLOYMENT.md            ← Web/Docker guide (post-Tier 1)
└── DOCUMENTATION_MAP.md     ← This file (navigation guide)
```

---

## Document Sizes

Understanding what to expect when reading each file:

**CLAUDE.md** (5-10 minutes)
- Overview document
- Quick reference
- Navigation guide
- Machine-readable YAML format

**QUICKSTART.md** (5-10 minutes)
- Setup instructions
- Task 1.1 complete specification
- What to expect from Claude Code

**TASKS.md** (scan in 5 minutes, reference as needed)
- All 32 tasks in YAML
- Use for looking up specific tasks
- Shows dependencies

**SPECS.md** (30-40 minutes, reference as needed)
- Detailed specifications for each component
- Read when implementing a specific component
- Configuration profiles explained

**DEPLOYMENT.md** (20-30 minutes, read after Tier 1)
- FastAPI setup instructions
- Docker configuration
- When/how to deploy

**README.md** (2-3 minutes)
- Project introduction
- Quick start commands
- Usage examples

---

## Document Formats

All documentation uses one of three formats for machine readability:

**YAML** (for structured data)
Used in: TASKS.md, SPECS.md, DEPLOYMENT.md
Easily parsed by Claude Code and other tools.

**Markdown with Code** (for guides)
Used in: QUICKSTART.md, README.md, DEPLOYMENT.md
Human-readable with syntax highlighting.

**Prose and YAML** (for navigation)
Used in: CLAUDE.md, DOCUMENTATION_MAP.md
Explains concepts and provides links.

---

## How to Use With Claude Code

1. Start Claude Code: `claude code`

2. Tell Claude Code to read a document:
   "Read QUICKSTART.md and implement Task 1.1"
   
3. Claude Code parses the YAML specifications and builds exactly what's described

4. Reference SPECS.md when you need detailed tier information:
   "According to SPECS.md Component 2, implement the Tier 1 boundary detector"

5. Check TASKS.md to see what comes next:
   "After Task 2.1, what's Task 2.2?"

---

## Quick Navigation by Role

**I am a developer implementing the project**
1. Read QUICKSTART.md (5 min)
2. Start with docs/TASKS.md Task 1.1
3. Reference SPECS.md as you build each component
4. Read DEPLOYMENT.md after Tier 1 is complete

**I am reviewing someone else's code**
1. Read CLAUDE.md for overview
2. Check SPECS.md for what they should have built
3. Reference TASKS.md to verify task completion
4. Use QUICKSTART.md to understand setup

**I want to understand the architecture**
1. Read CLAUDE.md (overview)
2. Read architecture.md (algorithm design)
3. Skim SPECS.md (component details)
4. Reference TASKS.md (task structure)

**I am deploying to production**
1. Verify Tier 1 complete with TASKS.md
2. Read DEPLOYMENT.md Phase W1 (web layer)
3. Read DEPLOYMENT.md Phase W2 (Docker)
4. Check DEPLOYMENT.md (monitoring section)

---

## Document Versioning

These documents will evolve as you build the project:

**When to update QUICKSTART.md**
- After Task 1.1 is implemented and tested
- When setup commands change

**When to update TASKS.md**
- Only if task structure changes (should be rare)
- Add notes about lessons learned

**When to update SPECS.md**
- As you encounter edge cases in implementation
- When tier specifications prove incorrect

**When to update DEPLOYMENT.md**
- After Phase W1 is deployed and tested
- When Docker configuration changes

**Never update CLAUDE.md**
- This is your stable reference
- Document architecture changes in SPECS.md instead

---

## Getting Help

If you're stuck:

1. Check this map to find the relevant documentation
2. Read that documentation carefully
3. Look for YAML examples in SPECS.md or DEPLOYMENT.md
4. Ask Claude Code to explain a concept:
   "Explain the three-tier system according to CLAUDE.md"

---

## Quick Links Within Documents

**In CLAUDE.md:**
- See section: "Project Structure Overview"
- See section: "Tier Development Sequence"

**In QUICKSTART.md:**
- See section: "Setting Up Your Environment"
- See section: "Task 1.1: Core Data Models"

**In TASKS.md:**
- Find task ID: Search for `task_id: 2.1`
- See dependencies: Look at `depends_on:` field

**In SPECS.md:**
- Find component specs: Search for `component: "Component X"`
- Find tier details: Search for `tier_1:`, `tier_2:`, `tier_3:`
- Find config profile: Search for `profiles:` section

**In DEPLOYMENT.md:**
- Find FastAPI setup: Search for "Phase W1: Basic Web Layer"
- Find Docker setup: Search for "Phase W2: Docker Packaging"
- Find library setup: Search for "Alternative: Library Installation"

---

## Questions This Map Answers

"Where do I start?"
→ QUICKSTART.md

"What's the complete task list?"
→ TASKS.md

"How do I implement this specific component?"
→ SPECS.md

"How do I deploy this to production?"
→ DEPLOYMENT.md

"How does everything fit together?"
→ CLAUDE.md

"What's the watermark algorithm?"
→ architecture.md

---

Last updated: As of project initialization. Update dates in specific documents as they change.
