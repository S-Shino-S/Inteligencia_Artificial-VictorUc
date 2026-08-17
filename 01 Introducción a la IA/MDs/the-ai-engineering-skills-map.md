This is part of a letter written by Andrew Ng on August 14, 2026, in The Batch (https://www.deeplearning.ai/the-batch/issue-366)

# The AI Engineering Skills Map

Based on an analysis of over 10,000 job postings; carrying out dozens of structured interviews with AI experts, hiring managers, and recruiters; gathering data through surveys; and synthesizing other online data, here are the four most important AI engineering skills:

- Building and deploying AI applications
- Software engineering fundamentals
- Using coding agents
- Shaping the build

You can informally think of our process as akin to clustering on a massive dataset of jobs and expert interviews to identify the most important skills, not just today but also in the near future.

## A note on terminology

I talk about **AI Engineering skills** rather than the “AI Engineer” role (someone whose job is to build AI systems), because the former is much broader. All developers today should know how to work with the cloud, and only a smaller number have a “Cloud engineer” title. Similarly, all developers — full-stack engineers, data engineers, DevOps engineers, machine learning engineers, and, yes, AI engineers — will need AI engineering skills.

## Building and deploying AI applications

The key difference between AI and non-AI applications is that the former has unpredictable outputs. When you prompt an LLM, you don’t know what you’ll get back. When you train a deep learning algorithm, you don’t know what prediction it will make on new examples. In contrast, traditional software behaves more predictably.

People who are skilled at building and deploying AI applications understand the building blocks of AI (such as LLMs, context engineering, RAG, agentic workflows, machine learning and deep learning) and, importantly, how to use statistical techniques to measure, steer, and govern AI systems so that they behave more predictably. A core skill in doing so is knowing how to drive disciplined evals and error analysis loops.

## Software engineering fundamentals

When you deeply understand how software works, you can build much more effectively. Engineering software requires making tradeoffs between cost, scalability, reliability, speed, and more. Security and privacy add further complexity.

Understanding software fundamentals allows you to recognize what tradeoffs even exist. This leads to better decisions in choosing your software stack, designing system architecture, designing your data store, testing, and so on. It also leads to much better outcomes than those for an inexperienced developer who vibe codes a solution without knowing the tradeoffs their coding agent is making — which will often be poor ones, because they don’t know what context to give their coding agent. Understanding software engineering fundamentals lets you make good tradeoffs by steering coding agents using the precise language of software engineering.

## Using coding agents

Using agentic coding effectively is now a key skill for every developer. When you have this skill, you have a good mental model for how agents work. You understand their limitations and how to work around them, and are able to quickly steer them — knowing how much to intervene and how much to leave them alone — to build robust software without wasting excessive time or tokens.

This requires your knowing how to manage a coding agent’s context, make tradeoffs between planning and execution, and help the agent autonomously close loops by providing verifiers or evals. You also need to know how to work with a clear spec (and when not to bother doing so), orchestrate multiple agents that work together, and avoid pitfalls that risk an agent messing up your production database. Because agentic coding is evolving quickly, using coding agents skillfully means not only knowing cutting-edge practices, but also having routines to keep trying new tools and evolve your workflows as best practices change.

## Shaping the build

Given a clear spec, coding agents are rapidly improving at delivering to it. Thus, our work as engineers is shifting toward deciding what should be in the spec. Engineers should no longer expect to be given a pixel-perfect design and asked only to implement it. Instead, effective AI engineering requires having product sense and understanding business context and customer goals, so you can participate in shaping and driving the build.

AI also gives you the opportunity to take on greater ownership and agency than before. You can identify interesting problems and opportunities, and execute to take advantage of them in responsible ways. Taking advantage of this opportunity requires knowing how to drive projects forward. For example, knowing when to quickly build an MVP to take to users for testing, and when to slow down and take longer in order to build more carefully.
