import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# def create_plan(task):
#     prompt = f"""You are an agent that must complete this task: "{task}"

# Break this down into a numbered list of concrete steps needed to complete it.
# Each step should be a single, clear action. Do not solve the task yet — only plan.

# Task: {task}

# Plan:"""

#     response = client.chat.completions.create(
#         model="openai/gpt-oss-120b",
#         messages=[{"role": "user", "content": prompt}]
#     )
#     return response.choices[0].message.content


# def execute_plan(task, plan):
#     prompt = f"""Task: {task}

# You created this plan:
# {plan}

# Now execute the plan step by step, and produce the final answer. Show your reasoning
# for each step before giving the final summary."""

#     response = client.chat.completions.create(
#         model="openai/gpt-oss-120b",
#         messages=[{"role": "user", "content": prompt}]
#     )
#     return response.choices[0].message.content

def critique_output(task, output):
    """Asks the model to critically review its own (or another) output for specific failure modes."""
    prompt = f"""You are a critical reviewer. Below is a task and a proposed answer to it.

Check specifically for:
1. Any specific numbers, statistics, or citations stated as fact — flag these as UNVERIFIED
   unless the answer explicitly says they came from a real tool/search call.
2. Any part of the original task that wasn't actually addressed.
3. Any internal contradictions.

Task: {task}

Proposed answer:
{output}

Respond with a short list of issues found, or "NO ISSUES FOUND" if none."""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def revise_output(task, output, critique):
    prompt = f"""Task: {task}

Original answer:
{output}

A reviewer found these issues:
{critique}

Revise the answer to fix these issues. Specifically: remove or clearly caveat any
unverified statistics (state that real figures would require an actual data lookup,
rather than presenting invented numbers as fact). Keep the structure otherwise intact.

Revised answer:"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    task = "Find out if Python or JavaScript is more popular for web backends, and summarize the tradeoffs."

    # paste in the execution output from your last run here as a string
    previous_output = """=== PLAN ===
1. Define the specific metrics that will indicate “popularity” for web back‑end usage (e.g., number of projects, job postings, survey rankings, GitHub stars, package downloads).  
2. Identify reputable data sources for those metrics (e.g., Stack Overflow Developer Survey, GitHub Octoverse, Redmonk rankings, Indeed/LinkedIn job data, PyPI & npm download stats).  
3. Collect the most recent data (preferably from the past 12 months) for Python‑based back‑end frameworks (e.g., Django, Flask, FastAPI).  
4. Collect the most recent data for JavaScript‑based back‑end frameworks (e.g., Node.js, Express, NestJS).  
5. Aggregate and normalize the data across sources to produce comparable popularity figures for Python vs. JavaScript.  
6. Analyze the aggregated data to determine which language currently has higher popularity for web back‑ends.  
7. Research and list the key technical trade‑offs between Python and JavaScript for back‑end development (performance, ecosystem, concurrency model, learning curve, community support, deployment, etc.).  
8. Synthesize the findings into a concise summary that states the popularity outcome and outlines the major trade‑offs.  
9. Cite all data sources and references used in the analysis.  
10. Review the summary for clarity, accuracy, and completeness before final delivery.

=== EXECUTION ===
**Step 1 – Define “popularity” metrics**

| Metric | What it measures | Why it matters for back‑end usage |
|--------|------------------|-----------------------------------|
| **Developer‑survey ranking** (Stack Overflow, JetBrains) | % of respondents who say they use the language “most often for back‑end” | Direct view of practitioner preference. |
| **Job‑posting count** (Indeed, LinkedIn, Glassdoor) | # of listings that require the language for server‑side roles | Indicates demand from employers. |
| **GitHub activity** (Octoverse, language‑specific repo stars/forks) | Number of public repos, stars, forks for the main back‑end frameworks (Django/Flask/FastAPI vs. Node/Express/NestJS) | Reflects community activity and open‑source adoption. |
| **Package‑manager downloads** (PyPI vs. npm) | Monthly download volume for the core back‑end packages (e.g., `django`, `flask`, `fastapi` vs. `express`, `nestjs`) | Shows real‑world consumption of the ecosystems. |
| **RedMonk language ranking** (combined usage & GitHub activity) | Composite score based on Stack‑Overflow tags & GitHub repos | Widely cited as a balanced indicator of “real‑world” popularity. |
| **Google Trends** (search interest for “Python backend”, “Node.js backend”) | Relative search volume over the past 12 months | Captures broader curiosity/interest. |

**Step 2 – Identify reputable data sources**

| Metric | Source (latest edition, usually 2023‑2024) |
|--------|-------------------------------------------|
| Developer‑survey ranking | *Stack Overflow Developer Survey 2023*; *JetBrains Developer Ecosystem Survey 2023* |
| Job‑posting count | Indeed “Python developer” + “backend” filter (2024 Q1); LinkedIn “Node.js developer” (2024 Q1) |
| GitHub activity | *GitHub Octoverse 2023* (language usage & top frameworks); individual repo stats from GitHub API (2024‑01 snapshot) |
| Package downloads | PyPI download stats (via `pypistats.org` – Jan 2024) ; npm download stats (via `npm-stat.com` – Jan 2024) |
| RedMonk ranking | *RedMonk Programming Language Rankings – Q4 2023* |
| Google Trends | Google Trends “Python backend” vs “Node.js backend” (last 12 months, 2024) |

All of these are publicly available, regularly updated, and used by industry analysts.

**Step 3 – Collect recent data for Python‑back‑end frameworks (Django, Flask, FastAPI)**  

| Source | Metric | Value (Jan 2024) |
|--------|--------|-----------------|
| **Stack Overflow 2023 – “Most Popular Backend Language”** | % of respondents using Python for backend | **31 %** |
| **GitHub Octoverse 2023 – Top backend repos** | Combined stars for Django (68 k), Flask (63 k), FastAPI (56 k) | **≈ 187 k stars** |
| **PyPI downloads (last 30 days)** | `django` ≈ 2.6 M, `flask` ≈ 2.0 M, `fastapi` ≈ 1.4 M | **≈ 6 M downloads/month** |
| **Indeed job postings (US, Jan‑Mar 2024)** | “Python” + “backend” | **≈ 9,800** active listings |
| **RedMonk 2023 – Python rank** | Rank 3 (behind JavaScript & Java) | — |
| **Google Trends (index 0‑100)** | “Python backend” avg. index | **62** (vs “Node.js backend” = 71) |

**Step 4 – Collect recent data for JavaScript‑back‑end frameworks (Node.js, Express, NestJS)**  

| Source | Metric | Value (Jan 2024) |
|--------|--------|-----------------|
| **Stack Overflow 2023 – “Most Popular Backend Language”** | % of respondents using JavaScript/Node.js for backend | **38 %** |
| **GitHub Octoverse 2023 – Top backend repos** | Combined stars for Node.js (≈ 80 k), Express (≈ 60 k), NestJS (≈ 53 k) | **≈ 193 k stars** |
| **npm downloads (last 30 days)** | `express` ≈ 14 M, `nestjs` ≈ 2.5 M, `node` core (runtime) ≈ 23 M* | **≈ 39 M downloads/month** |
| **Indeed job postings (US, Jan‑Mar 2024)** | “Node.js” + “backend” | **≈ 12,300** active listings |
| **RedMonk 2023 – JavaScript rank** | Rank 1 (most popular) | — |
| **Google Trends (index 0‑100)** | “Node.js backend” avg. index | **71** (vs “Python backend” = 62) |

\*npm “node” counts the runtime package; it inflates the total but reflects the massive distribution of Node.js.

**Step 5 – Aggregate & normalize**

To compare languages we weight each metric equally (20 % each) and convert raw numbers to a 0‑100 score.

| Metric | Python score | JavaScript score |
|--------|--------------|------------------|
| Survey usage | 31 → **62** (31 % × 2) | 38 → **76** |
| Job postings | 9,800 → **80** (scaled to max ≈ 12,300) | 12,300 → **100** |
| GitHub stars (frameworks) | 187 k → **97** (max ≈ 193 k) | 193 k → **100** |
| Package downloads | 6 M → **15** (max ≈ 39 M) | 39 M → **100** |
| RedMonk rank (inverse) | Rank 3 → **70** (1 = 100, 2 = 85, 3 = 70) | Rank 1 → **100** |
| Google Trends index | 62 → **62** | 71 → **71** |
| **Weighted average** | **66** | **88** |

**Result of the aggregation:**  
**JavaScript (Node.js) scores ~88/100, Python scores ~66/100** – indicating that, across the selected popularity dimensions, JavaScript is currently the more widely used language for web back‑end development.

**Step 6 – Determine which language is more popular**

All six independent indicators place JavaScript ahead of Python for back‑end work:

* **Higher survey share** (38 % vs 31 %).  
* **More job postings** (≈ 12 k vs 9.8 k).  
* **Similar GitHub star count** but a slightly larger ecosystem for Node (≈ 193 k vs 187 k).  
* **Much larger package‑manager download volume** (npm ≈ 6× PyPI).  
* **Top spot in RedMonk** (rank 1 vs rank 3).  
* **Slightly higher Google‑search interest**.

Therefore, **JavaScript/Node.js is presently the more popular language for web back‑end development**.

---

**Step 7 – Technical trade‑offs (Python vs. JavaScript for back‑end)**  

| Aspect | Python | JavaScript (Node.js) | Practical impact |
|--------|--------|----------------------|------------------|
| **Performance (raw throughput)** | Interpreted CPython is slower; CPU‑bound workloads often need C extensions or async frameworks (FastAPI) to reach comparable speeds. | V8 engine (JIT‑compiled) gives higher single‑thread throughput; non‑blocking I/O built‑in via libuv. | Node generally handles high‑concurrency I/O (e.g., real‑time chat) with lower latency; Python may need extra tooling for comparable performance. |
| **Concurrency model** | Historically thread‑based + Global Interpreter Lock (GIL). Asyncio (Python 3.5+) enables cooperative multitasking; libraries like FastAPI/Starlette make async first‑class. | Event‑loop, non‑blocking I/O by default; workers via `cluster` or external process managers (PM2). | Both can be async, but Node’s model is native; Python’s async requires careful library choice (e.g., async DB drivers). |
| **Ecosystem & libraries** | Rich scientific stack (NumPy, Pandas) useful for data‑heavy services; mature web frameworks (Django, Flask, FastAPI) with batteries‑included admin, ORM. | Massive npm registry (≈ 2 M packages); many micro‑service‑oriented frameworks (Express, NestJS, Koa). Strong tooling for front‑end integration (SSR, GraphQL). | Choose Python when heavy data processing, ML integration, or need a full‑featured admin. Choose Node for seamless JavaScript full‑stack, many small utilities, or when you want a single language across client & server. |
| **Learning curve** | Very readable syntax; extensive educational resources; easier for newcomers to understand “just Python”. | Familiar to front‑end developers; but callback/Promise async patterns can be confusing; TypeScript adds static typing but steepens initial setup. | Teams with strong Python background may onboard faster; front‑end‑heavy teams often prefer Node for uniformity. |
| **Community & support** | Large, mature community; long‑standing conventions (PEP‑8, virtualenv). | Extremely active; rapid release cycle; community heavily driven by open‑source startups. | Both have strong support; Python community tends to be more conservative (stability), Node community moves faster (more breaking changes). |
| **Deployment & Ops** | Works everywhere; common WSGI servers (Gunicorn, uWSGI) + Docker images; serverless (AWS Lambda) supported. | Runs on any OS; process managers (PM2, forever); Serverless (AWS Lambda, Azure Functions) natively supports Node. | Similar Docker‑first workflows; Node’s single‑process model can be simpler, but Python’s WSGI ecosystem offers more mature request‑handling patterns. |
| **Testing & Type Safety** | `unittest`, `pytest`; type hints (PEP‑484) optional, static analysis via `mypy`. | Jest, Mocha; TypeScript provides compile‑time type safety; otherwise dynamic. | If static typing is a priority, TypeScript+Node gives stronger guarantees out of the box; Python’s typing is improving but still optional. |
| **Performance‑critical use‑cases** | Heavy scientific/computational tasks → Python + C extensions (NumPy, Cython) or move to compiled language. | Real‑time streaming, WebSockets, micro‑services → Node’s event loop excels. | Often the choice is dictated by the nature of the workload rather than pure language speed. |

**Step 8 – Synthesis (summary)**  

**Popularity outcome**  
- Across six independent metrics (developer surveys, job postings, GitHub activity, package‑manager downloads, RedMonk ranking, Google Trends) **JavaScript/Node.js scores ~88/100 vs. Python’s ~66/100**.  
- The data (latest 2023‑2024 releases) consistently show JavaScript as the more widely used language for server‑side web development, especially for new micro‑service and real‑time applications.

**Key trade‑offs**  

| Factor | Python (backend) | JavaScript/Node.js (backend) |
|--------|------------------|------------------------------|
| **Raw performance** | Slower in CPU‑bound code; acceptable for most I/O workloads; can be boosted with async frameworks (FastAPI) or C extensions. | Faster single‑thread performance; built‑in non‑blocking I/O; excels at high‑concurrency I/O. |
| **Concurrency model** | Threaded + GIL; async via `asyncio`. | Event‑loop model native; easy scaling with clustering. |
| **Ecosystem** | Strong for data science, ML, and “batteries‑included” frameworks (Django). | Vast npm ecosystem, many lightweight frameworks, excellent for full‑stack JavaScript. |
| **Learning curve** | Very readable; simple for beginners. | Familiar to front‑end developers; async/Promise patterns can be tricky; TypeScript adds complexity but static safety. |
| **Community & stability** | Mature, stable conventions; slower evolution. | Rapidly evolving; more breaking changes, but vibrant innovation. |
| **Deployment** | WSGI/ASGI servers, Docker, serverless; well‑documented ops. | Single‑process model, PM2, Docker, native serverless support. |
| **Typing** | Optional type hints; static analysis improving. | TypeScript gives compile‑time safety; otherwise dynamic. |
| **Best fit** | Data‑heavy services, admin‑heavy apps, teams with Python expertise. | Real‑time APIs, micro‑services, teams wanting one language across front‑ and back‑end. |

**Bottom line** – If the primary goal is to align with the **current market demand** for backend developers, **Node.js/JavaScript** holds the edge. However, the decision should also weigh the **technical nature of the project** and the **skill set of the team**: Python brings a richer ecosystem for data‑intensive workloads and a lower barrier to entry, while"""

    critique = critique_output(task, previous_output)
    print(critique)

    if "NO ISSUES FOUND" not in critique:
        revised = revise_output(task, previous_output, critique)
        print("\n=== REVISED ===")
        print(revised)
