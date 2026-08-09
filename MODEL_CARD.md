# Model card. Born Field collision hazard model

Model: `hazard_poisson_glm` v1 · **Owner:** Born Field ·
Status: research demonstrator, fitted on synthetic data

---

## 1. Intended use

**In scope, infrastructure prioritisation and network screening.**
Ranking road segments by expected collisions per vehicle-mile so that a highway
authority can direct engineering surveys, signal retiming, geometric
improvements, or lighting upgrades. The output answers *"where would a fixed
safety budget buy the most risk reduction?"* The unit of decision is a road
segment, and the intervention is a change to the road.

Also in scope, aggregate exposure modelling. Portfolio-level or
corridor-level risk for fleet operators, where the question is about a route or
a network and not about a person.

**Explicitly out of scope.** These are not cautions; they are uses this model
must not be put to.

| Out-of-scope use | Why |
| --- | --- |
| Individual insurance pricing or underwriting | Territory-based rating is tightly constrained in California (see §4). The model has no individual-level covariates and cannot distinguish a careful driver on a risky road from a risky driver. |
| Law-enforcement deployment or patrol targeting | Creates a measurement feedback loop that corrupts the model and concentrates enforcement on already over-policed areas (§3). |
| Individual driver scoring, hiring, or licensing decisions | The model estimates a property of *places*, not people. Attributing a road's hazard rate to a driver who uses it is an ecological fallacy. |
| Property valuation or lending | Same ecological problem, plus direct proximity to redlining. |
| Real-time safety-critical control | Not validated for latency, availability, or failure modes; predicts hourly aggregates, not instantaneous hazard. |

## 2. What the model estimates

Expected collisions per cell-window, on an exposure offset:

```
log E[N] = log(VMT) + b0 + b1·log(flow) + b_class + b_night + b_wet
```

`VMT = flow × road-miles × hours`. Because the offset already supplies one power
of flow, `b1` estimates **α − 1**, where α is the flow exponent. **α = 1 means
risk is entirely explained by exposure and the intensity field adds nothing.**

Outputs are a rate per 100M vehicle-miles and a probability of at least one
collision, each with a 95% interval propagated through the log link.

A rate is not a ranking of danger. Because α > 1, risk per vehicle-mile
*rises* with flow, so raw crashes-per-VMT ranks motorways above residential
streets even though motorways are safer per vehicle-mile at equal flow. Any
consumer comparing road classes must condition on flow. This is the single most
likely way to misread the output.

## 3. Known biases and failure modes

Crash-report bias is real, spatially structured, and not correctable here.
Collision records are a *reporting* process layered on the event process, and
reporting rates vary by neighbourhood with language access, immigration status,
insurance coverage, and trust in police. Under-reported areas will be scored as
safer than they are, and the model has no way to distinguish "few crashes" from
"few reported crashes". The pipeline keeps `CrashObservation.reported` and
`FlowObservation.imputed` as first-class fields specifically so this is
representable.

Measured on synthetic data: with reporting probability varying 0.4–1.0 across
space, the flow exponent survives (bias +0.005) but total observed events fall
30%, so *absolute* rate levels are biased downward in under-reported areas even
where the exponent is unaffected. Relative rankings within a
uniformly-reported area remain usable; comparisons *across* areas with different
reporting regimes do not.

Feedback loops. If scores drive enforcement deployment, enforcement
generates reports, and reports feed the next fit, the model converges on
wherever it was first pointed rather than on where risk is. That is why enforcement targeting is out of scope, not merely discouraged. The same
loop does *not* arise for infrastructure use: fixing a road changes the hazard,
which is the intended effect, and the model correctly learns the new rate.

Detector measurement error is the dominant technical fragility. Loop-detector
volumes are noisy and heavily imputed. Because observed flow enters both the
offset and the covariate, the *whole* exponent attenuates: α̂ = α·λ, where λ is
the reliability ratio of log flow. Critically, λ is governed by the *within-class*
variance of log flow (0.21 in our fixture), not its marginal variance (1.89), the class dummies absorb nine tenths of it. Measured effect:

| Detector noise (σ, lognormal) | Recovered α (true 1.35) | Interval covers truth |
| --- | --- | --- |
| 0 (clean) | 1.351 | 12/12 |
| 0.30 | 0.914 | 0/12 |
| 0.60 | 0.474 | 0/12 |

At σ = 0.3 the recovered exponent falls **below 1**, which would report that risk
per vehicle-mile *decreases* with flow, the opposite of the truth. A real
deployment on PeMS-grade data requires an errors-in-variables correction or
validated flow. This is pre-registered failure condition F5, and it fires.

Omitted covariates. Removing two road-class strata biases α by −0.16 with
zero interval coverage. Any real feed missing a risk factor the true process
uses will load that factor onto the flow exponent.

Aggregation. Fitting at twice the generating cell size leaves α roughly
unbiased (+0.007) but more than doubles held-out deviance (0.051 → 0.114). Cell
size must be recorded with every fit; it is not a free parameter.

Rare-event overdispersion is nearly undetectable. Injecting gamma-mixture
overdispersion at the rate level changed neither the point estimate nor interval
coverage materially, and a negative-binomial fit gave the same answer as
Poisson. With events this rare, Poisson sampling noise swamps rate-level extra
variance. This is a limitation of the *diagnostic*, not evidence that
overdispersion is absent.

## 4. Regulatory context

Geographic risk scoring sits close to two heavily-regulated practices, and the
distance from each is a design property, not a disclaimer.

California Proposition 103 and auto rating factors. California constrains
private passenger auto rating: the mandatory factors (driving record, annual
mileage, years of driving experience) must carry more weight than optional ones,
and territorial rating is limited accordingly. A geographic hazard score of this
kind is not usable as a primary individual rating factor in California, and
would need review by a licensed actuary and the Department of Insurance before
entering any rating plan. Treat this paragraph as orientation, not legal advice:
a buyer's compliance team must make the determination.

Redlining and disparate impact. Geographic scores correlate with race and
income whether or not those variables are used, because residential segregation
makes geography a proxy. A model that is facially neutral can still produce
disparate impact. Two mitigations are structural here: the model estimates risk
*per vehicle-mile*, so it does not simply rediscover population density; and the intended use directs spending *toward* higher-risk
areas instead of charging them more. If a deployment inverts that direction, using the score to raise prices or withdraw service, the disparate-impact
analysis is entirely different and must be redone.

Predictive policing. The enforcement-targeting exclusion in §1 exists because
the feedback loop in §3 is well documented in that literature. Infrastructure
prioritisation does not share the mechanism.

## 5. Evaluation

Validation is spatially blocked cross-validation plus a forward-in-time holdout.
Random k-fold is not used and is not offered: spatial autocorrelation lets a
model interpolate between neighbouring cells, which no deployment ever gets to
do. The repository demonstrates the resulting optimism.

Primary metric is held-out Poisson deviance. PR-AUC and hit-rate @ top-5% are
secondary and reported per fold, since PR-AUC is prevalence-dependent and not
comparable across folds with differing crash density.

Pre-registered failure conditions F1–F5 were committed before any model existed
(see README and `git log`). F1–F4 pass; **F5 fires** and is reported above.

## 6. Data

Shipped results come from a synthetic generator sampling a known hazard law, over
a procedurally generated road network with realistic topology. **No real
collision data is used, and no number in this repository describes a real place.**
The generated network is a stand-in: `scripts/fetch_osm_extract.py` swaps in a
real OSM extract without changing anything downstream.

Adapters for Caltrans PeMS (flow) and TIMS/SWITRS (collisions) are documented
stubs. Anyone connecting them inherits every bias in §3 plus the licensing and
privacy terms of those sources.

## 7. Maintenance

Refit when the fitted time window no longer covers the query period; the service
refuses instead of extrapolating. Monitor the refusal rate by reason, a rise
in `outside_fitted_time_range` or `extrapolation_beyond_support` in one region is
the earliest signal that fitted support has drifted from live traffic. Refusals
are persisted alongside estimates for exactly this purpose.

## 8. Contact and escalation

Report suspected disparate impact, feedback-loop concerns, or out-of-scope
deployment through the repository's issue tracker. A deployment that touches
individual pricing or enforcement should be treated as out of scope until
reviewed, not merely flagged.
