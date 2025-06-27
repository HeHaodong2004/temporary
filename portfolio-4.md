---
title: "Generalized CBS-based Multi-Agent Path Planning Framework"
excerpt: "<br/><img src='/images/cbs.png'>"
collection: portfolio
---

![Generalized CBS MAPF](/images/mixed_cbs.gif)

### Overview
The field of **multi-agent path finding (MAPF)** has long relied on specialized methods tailored to gridworld environments, often limiting its applicability to more complex and dynamic domains. **Conflict-Based Search (CBS)** has emerged as a leading approach to tackle multi-agent coordination challenges by decomposing the problem into independent low-level plans and resolving conflicts through constraint-based replanning.

However, CBS has historically been perceived as limited to **gridworld-based motion**. This perception not only narrows the method’s reach but also underestimates its potential as a **generalizable hierarchical framework** capable of harnessing the strengths of a variety of low-level solvers. Exploring this flexibility could unlock new possibilities for robust and efficient path planning in diverse scenarios, from structured indoor environments to unstructured outdoor terrains, and across heterogeneous robot platforms.

---

### Focus Project: CBS with Diverse Low-Level Solvers
Our project **breaks new ground** by showcasing how CBS can be extended beyond its traditional confines, integrating a **suite of diverse low-level solvers** within a unified framework. Concretely, we combine CBS as the high-level planner with:

✅ Sampling-Based Solvers  
✅ Heuristic Search Solvers  
✅ Optimization Solvers  
✅ Reinforcement Learning Solvers  
✅ Diffusion Solvers  

This hybrid framework demonstrates CBS’s versatility in managing **heterogeneous footprints** (e.g., different robot shapes or dynamic models) and **heterogeneous low-level strategies**. Beyond simply adapting CBS to each of these domains separately, our work highlights how CBS can **simultaneously** orchestrate these solvers within the same environment, dynamically selecting the most suitable low-level planner for each conflict scenario.

Key contributions of our project include:

- **Unified demonstration** of CBS working with multiple low-level solvers  
- **Handling of heterogeneous agent models** and dynamic footprints  
- **User-friendly system design** that simplifies future extensions and experimentation  
- **Performance enhancements** in low-level planners based on real-world insights  

Inspired by works like **db-CBS (2023)**, we aim to push this idea further by building an open, extensible platform that showcases the true generality of CBS. This project marks a significant step toward making CBS accessible and impactful for a wider range of robotics and multi-agent coordination applications.
