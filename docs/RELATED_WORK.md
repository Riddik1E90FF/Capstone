# Related Work

**Neuroadaptive Curriculum Shaping for Reinforcement Learning**

- **Status:** Living document. Written to be folded into the capstone report's
  related-work chapter.
- **Date:** 2026-11-01
- **Scope:** Positions this project against four literatures — curriculum
  learning in RL, EEG spectral indices of workload, human-in-the-loop RL, and
  passive BCI — plus the statistical-practice literature that the analysis plan
  in `PREREGISTRATION.md` §9 rests on.
- **Binding documents:** `docs/PREREGISTRATION.md` (Stage-1 locked) and
  `docs/DECISIONS.md`. Nothing in this document changes either. Where the
  literature raises a question about the locked design, it is recorded in
  §8, *Questions for Ryan*, and nowhere else.

**Citation policy for this document.** Every reference listed in §9 was checked
against an authoritative source during the session in which this document was
written: a publisher record, a conference proceedings page, an indexed
bibliographic record, or the article's own title page. Authors, year, title,
venue and a resolvable DOI or stable URL were confirmed in each case. Anything
that could not be confirmed to that standard was excluded and is listed in §10,
*Could not verify*, with the reason. Where a cited work's own reported interval
is ambiguous in the sources available, that ambiguity is stated in the text
rather than smoothed over.

**Language.** Following `DECISIONS.md` **D025**, this document does not describe
EEG as reading thoughts, identifying emotion, or objectively measuring cognitive
load — and it does not describe other authors' work that way either, even where
their own abstracts are less cautious. Spectral band power is referred to as an
EEG-derived feature; the quantity it is taken to index is referred to as an
estimated or candidate correlate.

---

## 1. Curriculum learning in reinforcement learning

### 1.1 The ordering premise and its uneven empirical support

The idea that a learner benefits from encountering training material in a
non-arbitrary order predates deep learning. Elman (1993) showed that a recurrent
network could acquire a grammar with embedded clauses only when its early
training was restricted — "the importance of starting small" — and Bengio,
Louradour, Collobert and Weston (2009) generalised the observation into
*curriculum learning*, framing an easy-to-hard ordering as a continuation method
that smooths a non-convex optimisation landscape. That framing is the common
ancestor of everything in this section, including the present project.

What is easy to miss is how contested the empirical case has remained in the
supervised setting where it was first made. Weinshall, Cohen and Amir (2018)
established theoretical conditions under which an easy-to-hard ordering speeds
convergence, and — importantly for this project — obtained their difficulty
ordering *externally*, by transfer from a network pre-trained on a different
task, rather than from the learner being trained. Hacohen and Weinshall (2019)
extended this with explicit pacing functions and reported both faster learning
and improved final test accuracy on CIFAR-10, CIFAR-100 and ImageNet subsets,
again scoring difficulty either by a separate teacher network or by
bootstrapping.

Against this, Wu, Dyer and Neyshabur (2021) ran a large systematic study and
concluded that on standard benchmarks curricula offer only marginal benefits.
Their decisive control is the one that matters most here: randomly ordered
samples, with the *training-set size* grown over time, matched or beat both
curricula and anti-curricula. They attribute the apparent benefit of curriculum
learning to the dynamic training-set size rather than to the ordering itself.
Curricula did help in two specific regimes — a limited training budget, and
noisy labels — and anti-curricula did not help in either.

This is the single most important piece of prior work for calibrating
expectations about the present experiment, and it cuts against the project's
primary hypothesis in its general form. It also supplies an argument for a
control the pre-registration already contains. `SHUFFLE` permutes difficulty
*values* across configurations while preserving their marginal distribution
(`PREREGISTRATION.md` §4), which is precisely the manipulation that isolates
"ordering information" from "distribution over difficulty" — the distinction Wu
et al. show to be doing the work. A `NEURO ≈ SHUFFLE` outcome would be the
expected result under their account, and the pre-registration's M2 mechanism
already names it.

### 1.2 Automatic curriculum generation is overwhelmingly learner-centric

The RL curriculum literature, surveyed comprehensively by Narvekar, Peng,
Leonetti, Sinapov, Taylor and Stone (2020), has converged on signals derived
from the learner itself. Graves, Bellemare, Menick, Munos and Kavukcuoglu (2017)
treat task selection as a non-stationary multi-armed bandit whose reward is a
measure of the learner's own learning progress. Matiisen, Oliver, Cohen and
Schulman (2020) formalise the teacher–student setting, with a teacher policy
selecting sub-tasks in which the student is currently improving fastest.
Florensa, Held, Geng and Abbeel (2018) generate goals of intermediate difficulty
for the current policy using a generative adversarial objective, and Portelas,
Colas, Hofmann and Oudeyer (2020) do the equivalent over a continuously
parameterised environment space.

The strongest contemporary methods make the learner-centric commitment
explicit. Jiang, Grefenstette and Rocktäschel (2021) score environment levels by
a proxy for the agent's own value-estimation error and replay high-scoring
levels preferentially; the gains come from that scoring function, not from
non-uniform sampling as such. Unsupervised environment design (Dennis, Jaques,
Vinitsky, Bayen, Russell, Critch and Levine, 2020) casts curriculum generation
as a game in which an adversary proposes environments that maximise the
*regret* of the student relative to an antagonist, and Parker-Holder, Jiang,
Dennis, Samvelyan, Foerster, Grefenstette and Rocktäschel (2022) combine
regret-based selection with evolutionary editing of levels.

Two consequences follow for this project, and they should be stated plainly in
the report rather than discovered by an examiner.

First, an externally sourced ordering is *structurally less adaptive* than
anything in this paragraph. The EEG-derived index yields one fixed rank over 25
configurations, measured once from one human, and that rank does not update as
the agent learns. Every method above adjusts to the learner's changing
competence. The project's `PERF` condition — the agent's own running
per-configuration success rate — is the member of the seven that is closest in
spirit to the state of the art, and the literature's prior should therefore be
that `PERF` is a *strong* competitor rather than a convenience baseline. That is
exactly the comparison the pre-registration designates as the justification test
(H2).

Second, the project's contribution cannot be "a better curriculum method." It is
narrower and cleaner than that: an attempt to find out whether an ordering
obtained from a human's physiological response to a task carries information
about that task that is usable by a different kind of learner. Where the
externally sourced ordering fits, relative to the self-generated ones, is as a
*prior over task difficulty* available before the learner has produced any data
at all — which is the one thing regret- and progress-based methods cannot
supply, because they require the learner to have already encountered the task.

### 1.3 Curricula sourced from humans

A small literature asks whether humans supply good orderings. Khan, Mutlu and Zhu
(2011) had people teach a one-dimensional threshold concept to a robot and found
three distinct teaching strategies, *only one* of which is consistent with the
curriculum-learning principle. Humans, in other words, do not reliably generate
curriculum-shaped teaching sequences even when explicitly asked to teach.

Peng, MacGlashan, Loftin, Littman, Roberts and Taylor (2018) is the closest
prior work in the human-sourced-curriculum line, and arguably the closest
prior work to this project overall. Non-expert participants chose sequences of
simpler training tasks for a simulated agent in a Sokoban-like domain before it
faced a target task. The human-designed curricula helped the agent learn the
target task with less feedback than learning from scratch, despite participants
receiving no feedback on curriculum quality; the authors further identified
design principles in the human choices and showed that encoding those principles
in the algorithm improved performance again. Zeng, Duan, Li, Ferrara, Pinto, Kuo
and Nikolaidis (2022) move this into an interactive setting, with a human
adjusting task difficulty while watching an agent train.

Most directly relevant, and very recent: Solbach and Tsotsos (2025) order a
curriculum for a 3D visuospatial same–different task using *human accuracy* from
a prior psychophysics experiment, reordering the lesson sequence accordingly.
Their human-informed curriculum outperformed a naïve curriculum, with standard
PPO, behavioural cloning and GAIL failing on the task outright. This is positive
evidence that a human-derived difficulty ordering can transfer usefully to an
artificial learner.

It is also, read carefully, evidence about the wrong condition. Their ordering
signal is behavioural accuracy — which is this project's `BEHAV` condition, not
`NEURO`. The nearest published success for human-sourced curriculum ordering
therefore supports the premise behind H1 while simultaneously strengthening the
competitor that H2 must beat. This is discussed in §7.

---

## 2. EEG spectral indices of task difficulty and workload

### 2.1 What the band-power literature establishes

Two spectral features recur across four decades of workload research. Frontal
midline theta (roughly 4–8 Hz, maximal over Fz) tends to *increase* with task
demand; alpha (roughly 8–13 Hz, maximal posteriorly) tends to *decrease*.
Klimesch (1999) reviewed the evidence that alpha and theta oscillations covary
with cognitive and memory performance and argued for functionally distinct
sub-bands within alpha. Pfurtscheller and Lopes da Silva (1999) set out the
basic principles of event-related desynchronisation and synchronisation, the
mechanism conventionally invoked to explain why alpha power falls when a cortical
area is engaged. Gevins, Smith, Leong, McEvoy, Whitfield, Du and Rush (1998)
showed that multivariate EEG features could track working-memory load during
computer-based tasks well enough to classify load level within participants.
Cavanagh and Frank (2014) review frontal midline theta specifically, locating
its likely generators in mid-cingulate and pre-supplementary motor cortex and
arguing that it indexes the *need for control* — elicited by novelty, conflict,
error and punishing feedback. Their review is notably careful about limits: they
state that frontal theta does not always predict behavioural adjustment and that
its specific information content remains unresolved.

Applied work is correspondingly hedged. Borghini, Astolfi, Vecchiato, Mattia and
Babiloni (2014) review neurophysiological measurement in pilots and drivers and
treat the EEG indices as candidate correlates requiring task-specific
validation. Brouwer, Hogervorst, van Erp, Heffelaar, Zimmerman and Oostenveld
(2012) estimated workload from spectral power and ERPs in the n-back task and
are explicit that within-participant classification performance does not license
a general-purpose workload meter.

### 2.2 The meta-analytic effect sizes this project's index is built on

The index specified in `PREREGISTRATION.md` §7 is a fixed, literature-based
spectral formula with no trained parameters, weighted by the pooled effect sizes
reported by Chikhi, Matton and Blanchet (2022). That meta-analysis synthesised
45 effect sizes from 24 studies and 723 participants, and its headline results
are:

| Band | Pooled effect | Reported interval |
|---|---|---|
| Theta (4–8 Hz) | g = 0.68 | [0.41, 0.95] |
| Alpha (8–12 Hz) | g = −0.25 | printed as [−0.45, 0.04] in the published abstract |
| Beta (12–30 Hz) | g = 0.50 | [0.21, 0.79] |

The authors conclude that theta — particularly frontal theta — is the best of
the three as an index of cognitive workload, and that the relationships for
alpha and beta are "less straightforward." That ordering is faithfully reflected
in the project's weighting: theta primary, alpha secondary and down-weighted.

Three things about this table deserve to be said precisely, because the index is
the project's most load-bearing scientific commitment.

**The theta effect is moderate, not large, and it is a between-condition
average.** A pooled g of 0.68 across heterogeneous workload manipulations says
that theta reliably shifts in a known direction when demand increases. It does
not establish that theta power will separate 25 closely spaced configurations of
a single task within one session for one participant. That is a considerably
harder measurement problem than any study in the pool was designed to answer,
and the project's Gate 3 is the right place to find out whether it is possible
here. Nothing in Chikhi et al. (2022) licenses the assumption in advance.

**The alpha interval, as printed in the published abstract available this
session, includes zero.** The abstract describes the alpha effect as significant,
yet the interval is rendered as [−0.45, 0.04]. Given that the theta and beta
intervals are symmetric about their point estimates, symmetry would put alpha's
upper bound near −0.05, which suggests a lost minus sign somewhere between the
journal and the indexing services. The honest position is that the sign of
alpha's upper bound could not be resolved from the sources reachable this
session. This matters, and is raised in §8.

**Beta's pooled effect is larger in magnitude than alpha's, and the project's
index contains no beta term.** This is not a defect — a two-term index is
defensible, and `DECISIONS.md` **D023** leaves the feature set open — but the
report will be stronger if it states the exclusion and the reason for it rather
than leaving a reader to notice that the second-largest effect in the cited
meta-analysis was dropped while the smallest was kept.

### 2.3 What these measures do not establish

None of the work in this section supports the claim that an EEG band-power ratio
measures cognitive load in the sense of recovering a participant's subjective
experience. What the literature supports is narrower and sufficient for this
project: certain spectral features shift in reproducible directions when task
demand is manipulated, within a participant, under controlled recording
conditions. The project's use of the index — rank 25 configurations and nothing
more (`PREREGISTRATION.md` §6) — is well matched to that level of evidential
support, and the deliberate refusal to convert the index into a calibrated
difficulty magnitude is a genuine methodological strength that the report should
claim explicitly.

Two instruments in the project's protocol are inherited from this literature and
should be cited as such. The 1–9 subjective workload rating per block comes from
the scale used by Lim, Sourina and Wang (2018) in the STEW dataset, where 1–3,
4–6 and 7–9 denote low, moderate and high workload. The development fixture for
the EEG pipeline is the EEGMAT recording set of Zyma, Tukaev, Seleznov, Kiyono,
Popov, Chernykh and Shpenkov (2019); `DECISIONS.md` **D032** records that its
rest-versus-task contrast is confounded by eye state and is therefore void as a
workload validation, while the dataset remains usable as an implementation
fixture and for a Berger-effect positive control.

---

## 3. Human-in-the-loop reinforcement learning: where the human signal enters

The substantial literature on learning from humans is best organised not by the
*modality* of the human signal but by the *point at which it enters the learning
system*. On that axis this project sits somewhere almost nobody else does, and
the argument for its novelty should be made in those terms.

**Into the reward.** Knox and Stone (2009) introduced TAMER, in which a human's
scalar approval signal is modelled as a reward function and the agent acts
greedily with respect to the learned human-reward model; Warnell, Waytowich,
Lawhern and Stone (2018) scaled this to high-dimensional state spaces.
MacGlashan, Ho, Loftin, Peng, Wang, Roberts, Taylor and Littman (2017) showed
that human feedback is *policy-dependent* — people reward improvement rather than
absolute quality — and derived an advantage-based algorithm accordingly.

**Into a learned reward model, via comparisons.** Christiano, Leike, Brown,
Martic, Legg and Amodei (2017) elicit preferences over short trajectory segments
and fit a reward model to them, which is the design that subsequently became
standard practice for aligning large models.

**Into the policy, via demonstrations.** Abbeel and Ng (2004) recover a reward
function consistent with expert demonstrations and optimise against it, the
canonical inverse-reinforcement-learning formulation.

Arzate Cruz and Igarashi (2020) survey the interactive-RL design space across
these categories. What is striking when the survey is read with this project's
architecture in hand is that the categories are near-exhaustive for *reward* and
*policy* and nearly empty for *environment distribution*. The exception is the
human-sourced curriculum work of §1.3: Peng et al. (2018), Zeng et al. (2022)
and Solbach and Tsotsos (2025) all place the human's contribution in the
selection of training situations rather than in the reward or the policy, which
is the same structural position this project occupies.

The honest statement of distance is therefore this. The project's architectural
invariant — EEG influences only the distribution over environment
configurations, never the agent's observation, action or reward
(`PREREGISTRATION.md` §1) — is *not* unprecedented as a place to put human
information; Peng et al. (2018) got there first with explicit human choices, and
Solbach and Tsotsos (2025) got there most recently with human behavioural
accuracy. What appears to be unoccupied is the combination: a human signal that
is (i) physiological rather than chosen or behavioural, (ii) placed in the
environment distribution rather than the reward or policy, and (iii) evaluated
against controls that can separate human-specific information from mere
non-uniform structure. The claim the report can defensibly make is about the
combination, not about any one of the three.

---

## 4. Passive BCI and implicit human input

### 4.1 The paradigm

Zander and Kothe (2011) named and framed the category this project's EEG
component belongs to: a *passive* brain–computer interface derives information
from spontaneous brain activity to enrich a human–machine interaction, without
the user issuing voluntary commands. The distinction from active BCI is exactly
the distinction `CLAUDE.md` insists on for this project — there is no control
channel, and no intent is being decoded. Aricò, Borghini, Di Flumeri, Sciaraffa
and Babiloni (2018) review the field's move out of the laboratory and are
direct about the engineering obstacles that dominate real deployments: artifact
contamination under natural movement, signal non-stationarity, and the cost of
per-session calibration. Those are the same three problems `PREREGISTRATION.md`
§7 addresses with its artifact-control block, within-session standardisation,
and interleaved probe configurations.

Zander, Krol, Birbaumer and Gramann (2016) is the strongest demonstration in
this literature of a passive signal closing a loop. A cursor moved across a grid
toward a target with no explicit input: the system classified the operator's
event-related response to each movement, built a model of their directional
preferences from those classifications, and reinforced movements consistent with
the model. The signal was an EEG response the authors attribute largely to
medial prefrontal cortex. This is a reinforcement loop driven by a passive neural
signal — but the thing being optimised is the system's model of the user's
intent, not a separate learner's training distribution.

### 4.2 Physiological signals that have driven reinforcement learners

Three lines are close enough to require explicit differentiation.

Chavarriaga and Millán (2010) established that error-related potentials — the
EEG response when an observer perceives an incorrect outcome — can be decoded
single-trial and used as a learning signal for a BCI agent. Xu, Agarwal, Gupta,
Fekri and Sivakumar (2020) build on exactly this, decoding error-related
potentials from a human watching an agent and using the decoded labels to learn
an auxiliary reward function, which is combined with the sparse environment
reward. Akinola, Wang, Shi, He, Lapborisuth, Xu, Watkins-Valls, Sajda and Allen
(2020) decode a similar error signal and convert it, via supervised learning,
into a policy that is mixed into the agent's behaviour policy to steer early
exploration.

All three put the neural signal where this project explicitly refuses to put it:
into the reward (Xu et al.) or into the policy (Akinola et al.). They are also
*evaluative* — the human is judging the agent's behaviour — whereas this
project's human never observes the agent at all. The human plays the task
themselves; what is measured is their response to the task, not to the learner.
That difference is worth a paragraph in the report, because it is the cleanest
way to explain why the project's architecture is not a variant of EEG-based
reward shaping.

### 4.3 The biocybernetic loop, and a cautionary precedent

The oldest and, methodologically, the most relevant precedent is Pope, Bogart
and Bartolome (1995), who built a biocybernetic loop in which an EEG-derived
engagement index modulated the operator's own task — switching between manual
and automated control modes. Their index was beta / (alpha + theta), and they
validated candidate indices by an elegant criterion: under negative feedback a
good index should produce stable mode switching, under positive feedback it
should produce unstable switching. Two things follow. First, the structure
"EEG-derived index modulates the task environment" is thirty years old; what is
new here is modulating the environment *of a different, artificial learner*.
Second, the index family is a free parameter. Pope et al. chose a beta-weighted
ratio; this project chose theta with down-weighted alpha on meta-analytic
grounds; Ewing, Fairclough and Gilleade (2016) chose frontal theta plus upper
parietal alpha. The choice is defensible but it is a choice, and the report
should present it as one.

Ewing, Fairclough and Gilleade (2016) is the precedent that bears most directly
on whether this project's index can be expected to work, and it is a negative
result. They built an adaptive Tetris-style game whose block drop speed was
driven by a biocybernetic loop using frontal theta at Fz as an index of mental
effort and upper alpha (10.5–13 Hz) at right-parietal P4 as an index of game
demand — an index family nearly identical to the one specified in
`PREREGISTRATION.md` §7, and validated during their design process rather than
assumed. In evaluation, the EEG-adaptive versions did not clearly outperform the
comparison conditions: the only significant difference was that the conservative
adaptive version was rated more alerting than the liberal one, and a manual
control condition scored highest on immersion.

This should be cited in the report's limitations, not buried. The nearest
published attempt to drive an environment parameter from a theta-plus-alpha
index produced a null on its own primary outcome. It does not predict this
project's result — the outcome measure, the loop timescale and the learner are
all different, and a null there says nothing about whether the *ranking* carries
information — but it is the correct reference class for a prior, and it supports
the pre-registration's own judgement (§11) that the probability of `NEURO`
cleanly beating all baselines is below 50%.

---

## 5. Statistical practice in deep reinforcement learning

The analysis plan in `PREREGISTRATION.md` §9 — interquartile means, stratified
bootstrap intervals, permutation tests, all seeds reported — is not a stylistic
preference. It is the response to a documented methodological failure in this
field, and the report should say so.

Henderson, Islam, Bachman, Pineau, Precup and Meger (2018) demonstrated that
reported differences between deep RL algorithms were frequently not reproducible,
being sensitive to random seed, network architecture, reward scale and
implementation details that papers did not report. Colas, Sigaud and Oudeyer
(2018) applied statistical power analysis to deep RL comparisons and showed that
the seed counts then conventional were far too small to detect the effect sizes
being claimed; they give procedures for choosing a seed count from the variance
actually observed, which is precisely the structure of this project's Stage-2
amendment.

Agarwal, Schwarzer, Castro, Courville and Bellemare (2021) is the work the
analysis plan follows most closely. Their argument is that with few runs per
task, point estimates of aggregate performance — mean or median — carry
substantial and usually unreported statistical uncertainty, and that this has
produced discrepancies in published comparisons. Their recommendations are
interval estimates rather than point estimates, performance profiles to show
the distribution across runs, and robust aggregate statistics, specifically the
interquartile mean, which attains small uncertainty at modest run counts. The
pre-registration's choice to bootstrap the difference of interquartile means
rather than of raw means is a direct application.

Patterson, Neumann, White and White (2024) extend the methodological argument
from reporting to design, covering the choices — evaluation protocol,
hyperparameter selection, what counts as a run — that determine whether a
comparison means anything before any statistics are computed. Their framing
supports two decisions already made here that are easy to undervalue: the
requirement that all seven conditions share one frontier-sampler implementation
with the ordering source injected as a dependency (`PREREGISTRATION.md` §4), and
Gate 1a's demand that the from-scratch DQN be validated against a reference
implementation before it touches the experimental environment. Both exist to
ensure that an observed between-condition difference is attributable to the
independent variable rather than to implementation drift.

---

## 6. The gap

Stated as a gap rather than as a contribution:

**Curriculum learning in RL** has developed sophisticated machinery for
generating training distributions, and that machinery is almost entirely
learner-centric — progress, value-estimation error, regret. The question of
whether an ordering obtained from a *source outside the learning system* carries
usable information has been asked mainly in supervised learning, where it was
answered with a transfer-learning teacher (Weinshall et al., 2018; Hacohen and
Weinshall, 2019) and where the strongest systematic study found the ordering
contributes little once training-set size is controlled (Wu et al., 2021). It has
not been asked with a human physiological signal as the source.

**The EEG workload literature** has established, at the level of moderate pooled
effects across heterogeneous paradigms, that certain spectral features track
manipulated task demand (Chikhi et al., 2022). It has not asked whether the
*rank ordering* those features induce over a set of task configurations is
informative to anything other than a model of the human who produced it.

**Human-in-the-loop RL** has thoroughly explored the human signal as reward
(Knox and Stone, 2009; MacGlashan et al., 2017; Warnell et al., 2018),
as preference data (Christiano et al., 2017), and as demonstration (Abbeel and
Ng, 2004). The environment-distribution entry point has been explored with
explicit human task selection (Peng et al., 2018; Zeng et al., 2022) and with
human behavioural accuracy (Solbach and Tsotsos, 2025), but not with a
physiological measure.

**Passive BCI** has closed loops from neural signals to an environment for
thirty years (Pope et al., 1995; Ewing et al., 2016) and to an agent's reward or
policy more recently (Chavarriaga and Millán, 2010; Xu et al., 2020; Akinola et
al., 2020). In the first case the environment being modulated is the human's
own; in the second the neural signal reaches the learner directly. No work
located for this review modulates a separate artificial learner's training
distribution from a passively measured human signal while keeping that signal
out of the learner entirely.

What this project adds is therefore modest and specific: a single-subject,
pre-registered, falsification-oriented test of whether an untrained,
literature-weighted EEG-derived difficulty ranking over a fixed configuration
set orders an RL curriculum better than controls that hold its distributional
properties fixed (`SHUFFLE`), invert it (`REVERSE`), source the ordering from
the learner (`PERF`), source it from the human's behaviour (`BEHAV`), or use no
ordering at all (`RANDOM`, `STATIC`). The control set, rather than any component,
is what makes the question answerable. No claim of generality follows from one
participant and one small simulator, and the pre-registration is right to
foreclose one.

---

## 7. What the literature implies about whether the hypotheses are plausible

Collected here so the report can state a calibrated prior rather than an
optimistic one. This section argues from the literature; it does not propose
design changes.

### Evidence consistent with the project's premise

- **Peng et al. (2018).** Human-designed curricula helped an agent learn a target
  task with less feedback than learning from scratch, with non-expert designers
  receiving no quality feedback. Human-sourced orderings can carry usable
  information.
- **Solbach and Tsotsos (2025).** A curriculum ordered by human accuracy beat a
  naïve curriculum on a 3D visuospatial RL task where PPO, behavioural cloning
  and GAIL failed outright. The closest published success for the general
  premise.
- **Chikhi et al. (2022).** The direction and moderate magnitude of the theta
  effect (g = 0.68) support theta as the primary term of an untrained index,
  which is how the project uses it.
- **Weinshall et al. (2018); Hacohen and Weinshall (2019).** An externally
  sourced difficulty ordering — from a network trained on a different task — can
  produce measurable benefits, establishing that the ordering need not come from
  the learner.

### Evidence that cuts against it

- **Wu, Dyer and Neyshabur (2021).** The most systematic study of when curricula
  help concludes the benefit is marginal on standard benchmarks and attributable
  to dynamic training-set size rather than ordering. This is a direct argument
  for M2 or M3 over M1. Their two exceptions — limited training budget, noisy
  signal — are the regimes where ordering mattered, and whether this project sits
  in either is a property of the step budget and the environment's stochasticity
  rather than of the EEG index.
- **Ewing, Fairclough and Gilleade (2016).** A closed loop driven by frontal
  theta plus upper parietal alpha, modulating a game difficulty parameter, did
  not clearly outperform its comparison conditions. The nearest methodological
  precedent for the index produced a null on its own outcome measure.
- **Khan, Mutlu and Zhu (2011).** Only one of three observed human teaching
  strategies matched the curriculum-learning principle. Human-derived orderings
  are not reliably curriculum-shaped.
- **Dubey, Agrawal, Pathak, Griffiths and Efros (2018).** Human priors — object
  semantics, affordance, visual similarity — substantially aid human video-game
  performance while RL agents are largely indifferent to their removal. This is
  empirical support for the Moravec (1988) asymmetry that motivates M1-inv: the
  features that dominate human task difficulty need not be the features that
  dominate an agent's. It is simultaneously evidence against H1 and evidence for
  H5, which is exactly why having both in one directional family is a sound
  design.
- **The learner-centric curriculum literature** (Jiang et al., 2021; Dennis et
  al., 2020; Parker-Holder et al., 2022). The field's prior should be that a
  signal derived from the learner's own difficulty beats a static external rank.
  That makes `PERF` the most likely winner among the seven conditions and H2 the
  hypothesis most at risk.
- **Solbach and Tsotsos (2025), read against H2.** The nearest success used human
  *behavioural* accuracy. If a behavioural ordering is what transfers, `BEHAV`
  may match or beat `NEURO` — the pre-registered "M1 supported, H2 refuted"
  outcome. The pre-registration already names this as a legitimate finding.

### Summary of the prior

The literature supports the premise that *some* human-derived ordering can
transfer to an artificial learner, supports theta as the best available untrained
EEG workload correlate, and offers no precedent for the EEG-derived ordering
specifically outperforming free alternatives. The most probable outcomes on
current evidence are the pre-registration's M2 (structure, not human
information), M3 (null), or "M1 supported, H2 refuted" (EEG carries information
but no advantage over behavioural or agent-derived signals). All three are
pre-registered as reportable findings, which is the design's main protection.

---

## 8. Questions for Ryan

Raised by the reading, not acted on. None of these is a proposed design change;
each is a question whose answer belongs in the report, the deviations log, or
neither.

**Q1 — Resolve the alpha confidence interval in Chikhi et al. (2022).** The
published abstract, as rendered by every indexing service reachable this session,
gives alpha as g = −0.25, CI [−0.45, 0.04] while describing the effect as
significant. Symmetry with the theta and beta intervals implies the upper bound
should be about −0.05, so this is most likely a lost minus sign. It should be
checked against the full text before the report cites it, because the two
readings have different consequences: if the interval genuinely includes zero,
the secondary alpha term of the index rests on a pooled effect that did not reach
significance, and the pre-registered degraded mode in §7 (drop alpha, run on
frontal theta alone) stops being a hardware contingency and becomes arguably the
better-supported default. I have written the ambiguity into §2.2 rather than
resolving it by guess.

**Q2 — Why no beta term?** Chikhi et al. report beta at g = 0.50 [0.21, 0.79],
larger in magnitude and with a tighter interval than alpha's. The index uses
theta and alpha. This is a legitimate simplification and `DECISIONS.md` D023
leaves features open, but a reader of the final report will notice that the
second-largest pooled effect was excluded while the smallest was retained. A
sentence of rationale — scope, electrode coverage, beta's known contamination by
EMG during active movement, or simply that the two-term index was fixed before
the meta-analysis was read in this detail — would close the gap. Worth noting
that beta's EMG susceptibility is a particularly good reason given this project's
artifact constraints, if that is in fact the reason.

**Q3 — Alpha band definition: 8–13 Hz versus upper alpha.** §7 specifies
8–13 Hz for the alpha term. Chikhi et al. define alpha as 8–12 Hz. Klimesch
(1999) argues for functionally distinct lower and upper alpha sub-bands, and
Ewing et al. (2016) — the nearest methodological precedent — used upper alpha
specifically, 10.5–13 Hz at P4. If the band definition is still open, the
difference is worth a deliberate decision with a recorded reason rather than
being settled by whichever number was written first. If it is closed, the report
should say which convention was adopted and that the choice was made in advance.

**Q4 — Should a prior for `PERF` be recorded before results exist?** §3 records
a prior expectation for H3 ("least likely to show a clean effect") specifically
to prevent post-hoc reframing. No comparable prior is recorded for `NEURO` versus
`PERF`, yet the curriculum literature's clear expectation is that a
learner-derived ordering beats a static external one. Recording that expectation
now would strengthen the report's claim to have predicted rather than
rationalised its outcome. Stage 2 is reserved for seed count and smallest effect
size, so this would be a §12 clarification entry rather than an amendment — and
only if you judge it worth the entry.

**Q5 — Is the step budget a hidden moderator?** Wu et al. (2021) found ordering
effects appearing mainly under a *limited* training budget. The project's fixed
step budget is therefore not a neutral implementation detail: a null across all
seven conditions at one budget could be a budget-specific null. This is not an
argument for adding a budget sweep — that is scope creep against a seven-
condition design — but it is an argument for stating in the limitations that the
comparison is conditional on the chosen budget, and for recording how the budget
was chosen.

**Q6 — Ewing et al. (2016) belongs in the report's framing, not just its
bibliography.** It is the closest precedent for the index-plus-loop design and
it is a null. Citing it prominently costs nothing and pre-empts the obvious
examiner question ("has anyone tried this and did it work?"). It also supports
the pre-registration's stated sub-50% prior with an external reference rather
than only internal judgement.

**Q7 — Solbach and Tsotsos (2025) is recent enough to warrant a direct read.**
It is the nearest published success for human-ordered RL curricula and it
arrived in November 2025. Their ordering signal is human accuracy, which is this
project's `BEHAV` condition; how they handled the human-to-agent difficulty
mapping is likely to be directly useful at Gate 2, and their result is the best
available external evidence on the H1-versus-H2 tension.

---

## 9. References

All entries below were verified against an authoritative source during the
preparation of this document: authors, year, title, venue, and a resolvable DOI
or stable URL. 46 references.

Abbeel, P., & Ng, A. Y. (2004). Apprenticeship learning via inverse reinforcement learning. In *Proceedings of the Twenty-First International Conference on Machine Learning (ICML '04)*. ACM. https://doi.org/10.1145/1015330.1015430

Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A., & Bellemare, M. G. (2021). Deep reinforcement learning at the edge of the statistical precipice. *Advances in Neural Information Processing Systems*, 34, 29304–29320. https://arxiv.org/abs/2108.13264

Akinola, I., Wang, Z., Shi, J., He, X., Lapborisuth, P., Xu, J., Watkins-Valls, D., Sajda, P., & Allen, P. (2020). Accelerated robot learning via human brain signals. arXiv:1910.00682. https://arxiv.org/abs/1910.00682

Aricò, P., Borghini, G., Di Flumeri, G., Sciaraffa, N., & Babiloni, F. (2018). Passive BCI beyond the lab: current trends and future directions. *Physiological Measurement*, 39(8). https://doi.org/10.1088/1361-6579/aad57e

Arzate Cruz, C., & Igarashi, T. (2020). A survey on interactive reinforcement learning: design principles and open challenges. In *Proceedings of the 2020 ACM Designing Interactive Systems Conference (DIS '20)*, 1195–1209. https://doi.org/10.1145/3357236.3395525

Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). Curriculum learning. In *Proceedings of the 26th International Conference on Machine Learning (ICML '09)*, 41–48. ACM. https://doi.org/10.1145/1553374.1553380

Borghini, G., Astolfi, L., Vecchiato, G., Mattia, D., & Babiloni, F. (2014). Measuring neurophysiological signals in aircraft pilots and car drivers for the assessment of mental workload, fatigue and drowsiness. *Neuroscience & Biobehavioral Reviews*, 44, 58–75. https://doi.org/10.1016/j.neubiorev.2012.10.003

Brouwer, A.-M., Hogervorst, M. A., van Erp, J. B. F., Heffelaar, T., Zimmerman, P. H., & Oostenveld, R. (2012). Estimating workload using EEG spectral power and ERPs in the n-back task. *Journal of Neural Engineering*, 9(4), 045008. https://doi.org/10.1088/1741-2560/9/4/045008

Cavanagh, J. F., & Frank, M. J. (2014). Frontal theta as a mechanism for cognitive control. *Trends in Cognitive Sciences*, 18(8), 414–421. https://doi.org/10.1016/j.tics.2014.04.012

Chavarriaga, R., & Millán, J. del R. (2010). Learning from EEG error-related potentials in noninvasive brain–computer interfaces. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 18(4), 381–388. https://doi.org/10.1109/TNSRE.2010.2053387

Chikhi, S., Matton, N., & Blanchet, S. (2022). EEG power spectral measures of cognitive workload: A meta-analysis. *Psychophysiology*, 59(6), e14009. https://doi.org/10.1111/psyp.14009

Christiano, P., Leike, J., Brown, T., Martic, M., Legg, S., & Amodei, D. (2017). Deep reinforcement learning from human preferences. *Advances in Neural Information Processing Systems*, 30. https://proceedings.neurips.cc/paper/2017/file/d5e2c0adad503c91f91df240d0cd4e49-Paper.pdf

Colas, C., Sigaud, O., & Oudeyer, P.-Y. (2018). How many random seeds? Statistical power analysis in deep reinforcement learning experiments. arXiv:1806.08295. https://arxiv.org/abs/1806.08295

Dennis, M., Jaques, N., Vinitsky, E., Bayen, A., Russell, S., Critch, A., & Levine, S. (2020). Emergent complexity and zero-shot transfer via unsupervised environment design. *Advances in Neural Information Processing Systems*, 33. https://arxiv.org/abs/2012.02096

Dubey, R., Agrawal, P., Pathak, D., Griffiths, T., & Efros, A. (2018). Investigating human priors for playing video games. In *Proceedings of the 35th International Conference on Machine Learning*, PMLR 80, 1349–1357. https://arxiv.org/abs/1802.10217

Elman, J. L. (1993). Learning and development in neural networks: The importance of starting small. *Cognition*, 48, 71–99.

Ewing, K. C., Fairclough, S. H., & Gilleade, K. (2016). Evaluation of an adaptive game that uses EEG measures validated during the design process as inputs to a biocybernetic loop. *Frontiers in Human Neuroscience*, 10, 223. https://doi.org/10.3389/fnhum.2016.00223

Florensa, C., Held, D., Geng, X., & Abbeel, P. (2018). Automatic goal generation for reinforcement learning agents. In *Proceedings of the 35th International Conference on Machine Learning*, PMLR 80, 1515–1528. https://proceedings.mlr.press/v80/florensa18a.html

Gevins, A., Smith, M. E., Leong, H., McEvoy, L., Whitfield, S., Du, R., & Rush, G. (1998). Monitoring working memory load during computer-based tasks with EEG pattern recognition methods. *Human Factors*, 40(1), 79–91. https://doi.org/10.1518/001872098779480578

Graves, A., Bellemare, M. G., Menick, J., Munos, R., & Kavukcuoglu, K. (2017). Automated curriculum learning for neural networks. In *Proceedings of the 34th International Conference on Machine Learning*, PMLR 70, 1311–1320. https://proceedings.mlr.press/v70/graves17a.html

Hacohen, G., & Weinshall, D. (2019). On the power of curriculum learning in training deep networks. In *Proceedings of the 36th International Conference on Machine Learning*, PMLR 97, 2535–2544. https://proceedings.mlr.press/v97/hacohen19a.html

Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., & Meger, D. (2018). Deep reinforcement learning that matters. In *Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence*. https://arxiv.org/abs/1709.06560

Jiang, M., Grefenstette, E., & Rocktäschel, T. (2021). Prioritized level replay. In *Proceedings of the 38th International Conference on Machine Learning*, PMLR 139, 4940–4950. https://proceedings.mlr.press/v139/jiang21b.html

Khan, F., Mutlu, B., & Zhu, X. (2011). How do humans teach: On curriculum learning and teaching dimension. *Advances in Neural Information Processing Systems*, 24. https://proceedings.neurips.cc/paper/2011/hash/f9028faec74be6ec9b852b0a542e2f39-Abstract.html

Klimesch, W. (1999). EEG alpha and theta oscillations reflect cognitive and memory performance: A review and analysis. *Brain Research Reviews*, 29(2–3), 169–195. https://doi.org/10.1016/S0165-0173(98)00056-3

Knox, W. B., & Stone, P. (2009). Interactively shaping agents via human reinforcement: The TAMER framework. In *The Fifth International Conference on Knowledge Capture (K-CAP '09)*. https://www.cs.utexas.edu/~pstone/Papers/bib2html/b2hd-KCAP09-knox.html

Lim, W. L., Sourina, O., & Wang, L. P. (2018). STEW: Simultaneous task EEG workload data set. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 26(11), 2106–2114. https://doi.org/10.1109/TNSRE.2018.2872924

MacGlashan, J., Ho, M. K., Loftin, R., Peng, B., Wang, G., Roberts, D. L., Taylor, M. E., & Littman, M. L. (2017). Interactive learning from policy-dependent human feedback. In *Proceedings of the 34th International Conference on Machine Learning*, PMLR 70, 2285–2294. https://proceedings.mlr.press/v70/macglashan17a.html

Matiisen, T., Oliver, A., Cohen, T., & Schulman, J. (2020). Teacher–student curriculum learning. *IEEE Transactions on Neural Networks and Learning Systems*, 31(9), 3732–3740. https://doi.org/10.1109/TNNLS.2019.2934906

Moravec, H. (1988). *Mind Children: The Future of Robot and Human Intelligence*. Harvard University Press.

Narvekar, S., Peng, B., Leonetti, M., Sinapov, J., Taylor, M. E., & Stone, P. (2020). Curriculum learning for reinforcement learning domains: A framework and survey. *Journal of Machine Learning Research*, 21(181), 1–50. https://jmlr.org/papers/v21/20-212.html

Parker-Holder, J., Jiang, M., Dennis, M., Samvelyan, M., Foerster, J., Grefenstette, E., & Rocktäschel, T. (2022). Evolving curricula with regret-based environment design. In *Proceedings of the 39th International Conference on Machine Learning*, PMLR 162, 17473–17498. https://proceedings.mlr.press/v162/parker-holder22a.html

Patterson, A., Neumann, S., White, M., & White, A. (2024). Empirical design in reinforcement learning. *Journal of Machine Learning Research*, 25(318), 1–63. https://jmlr.org/papers/v25/23-0183.html

Peng, B., MacGlashan, J., Loftin, R., Littman, M. L., Roberts, D. L., & Taylor, M. E. (2018). Curriculum design for machine learners in sequential decision tasks. *IEEE Transactions on Emerging Topics in Computational Intelligence*, 2(4), 268–277. https://doi.org/10.1109/TETCI.2018.2829980

Pfurtscheller, G., & Lopes da Silva, F. H. (1999). Event-related EEG/MEG synchronization and desynchronization: Basic principles. *Clinical Neurophysiology*, 110(11), 1842–1857. https://doi.org/10.1016/S1388-2457(99)00141-8

Pope, A. T., Bogart, E. H., & Bartolome, D. S. (1995). Biocybernetic system evaluates indices of operator engagement in automated task. *Biological Psychology*, 40(1–2). https://doi.org/10.1016/0301-0511(95)05116-3

Portelas, R., Colas, C., Hofmann, K., & Oudeyer, P.-Y. (2020). Teacher algorithms for curriculum learning of deep RL in continuously parameterized environments. In *Proceedings of the Conference on Robot Learning*, PMLR 100, 835–853. https://proceedings.mlr.press/v100/portelas20a.html

Solbach, M. D., & Tsotsos, J. K. (2025). Boosting reinforcement learning in 3D visuospatial tasks through human-informed curriculum design. arXiv:2511.17595. https://arxiv.org/abs/2511.17595

Warnell, G., Waytowich, N., Lawhern, V., & Stone, P. (2018). Deep TAMER: Interactive agent shaping in high-dimensional state spaces. In *Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence*. https://ojs.aaai.org/index.php/AAAI/article/view/11485

Weinshall, D., Cohen, G., & Amir, D. (2018). Curriculum learning by transfer learning: Theory and experiments with deep networks. In *Proceedings of the 35th International Conference on Machine Learning*, PMLR 80, 5238–5246. https://proceedings.mlr.press/v80/weinshall18a.html

Wu, X., Dyer, E., & Neyshabur, B. (2021). When do curricula work? *International Conference on Learning Representations (ICLR 2021)*. arXiv:2012.03107. https://arxiv.org/abs/2012.03107

Xu, D., Agarwal, M., Gupta, E., Fekri, F., & Sivakumar, R. (2020). Accelerating reinforcement learning using EEG-based implicit human feedback. arXiv:2006.16498. https://arxiv.org/abs/2006.16498

Zander, T. O., & Kothe, C. (2011). Towards passive brain–computer interfaces: applying brain–computer interface technology to human–machine systems in general. *Journal of Neural Engineering*, 8(2), 025005. https://doi.org/10.1088/1741-2560/8/2/025005

Zander, T. O., Krol, L. R., Birbaumer, N. P., & Gramann, K. (2016). Neuroadaptive technology enables implicit cursor control based on medial prefrontal cortex activity. *Proceedings of the National Academy of Sciences*, 113(52), 14898–14903. https://doi.org/10.1073/pnas.1605155114

Zeng, Y., Duan, J., Li, Y., Ferrara, E., Pinto, L., Kuo, C.-C. J., & Nikolaidis, S. (2022). Human decision makings on curriculum reinforcement learning with difficulty adjustment. arXiv:2208.02932. https://arxiv.org/abs/2208.02932

Zyma, I., Tukaev, S., Seleznov, I., Kiyono, K., Popov, A., Chernykh, M., & Shpenkov, O. (2019). Electroencephalograms during mental arithmetic task performance. *Data*, 4(1), 14. https://www.mdpi.com/2306-5729/4/1/14

---

## 10. Could not verify

Excluded from §9. Listed so the exclusions are auditable and so that none is
silently reintroduced later from memory.

| Reference sought | Why it was wanted | What happened |
|---|---|---|
| Jensen & Tesche (2002), frontal theta increases with memory load, *European Journal of Neuroscience* 15, 1395–1399 | The canonical primary source for the theta-load relationship that is the index's primary term | Title, journal, volume, pages and DOI (10.1046/j.1460-9568.2002.01975.x) all confirmed, and the first author confirmed as O. Jensen. The **second author could not be confirmed** from any reachable record — every index returned the first author only. Excluded under the citation rule rather than completed from memory. The theta-load claim in §2 is carried by Chikhi et al. (2022) and Cavanagh & Frank (2014) instead. Worth reinstating once the article's own title page can be checked. |
| Xu et al., journal version in *Neurocomputing* (2021) | Preferred over a preprint for a substantive claim | The arXiv version (2006.16498) was fully verified, including the title-page footnote noting partial prior publication in an AAAI workshop. The *Neurocomputing* volume, issue, pages and DOI could not be confirmed. §9 cites the arXiv version only. |
| Akinola et al., ICRA 2020 proceedings record | Preferred over a preprint | Title and all nine authors verified from the authors' own PDF and the arXiv record. The **venue could not be confirmed** from any authoritative source reachable this session (a file path suggesting ICRA 2020 is not evidence). §9 cites the arXiv version and makes no venue claim. |
| Elman (1993) DOI and issue number | Completeness | Title, author, journal, volume, pages and year verified from the article's own first page. The DOI and issue number could not be confirmed, so neither is asserted. |
| Pope et al. (1995) page range | Completeness | Authors, year, title, journal, volume, issue and DOI verified via the NASA technical-reports record. The page range was not shown there and is not asserted. |
| Chikhi et al. (2022) alpha confidence-interval sign | Determines whether the index's secondary term rests on a significant pooled effect | Every reachable rendering of the published abstract prints [−0.45, 0.04], which contradicts the abstract's own description of the effect as significant and breaks the symmetry seen in the theta and beta intervals. Reported as an open ambiguity in §2.2 and raised as Q1 in §8. **Not resolved by inference.** |
| A direct precedent for an EEG-derived signal shaping a *separate* RL agent's training-environment distribution | Would have changed the gap statement in §6 | Searched for specifically. None found. Absence of a found precedent is not proof of absence, and §6 is worded accordingly. |
