# Nebius x NVIDIA Global AI Hackathon — Validated Requirements

Sources: [overview](https://nebiusglobalaihackathon.devpost.com/), [official rules](https://nebiusglobalaihackathon.devpost.com/rules), and [resources](https://nebiusglobalaihackathon.devpost.com/resources), checked 2026-09-29. The official rules control if the pages conflict and may change before the deadline.

## Dates

| Milestone | When |
|---|---|
| Submission period opens | 2026-08-26, 09:00 PDT |
| **Submission deadline** | **2026-10-30, 10:00 PDT** |
| Judging | 2026-12-01, 09:00 PST → 2026-12-15, 12:00 PST |
| Winners announced | On or around 2027-01-11, 12:00 PST |

Time from 2026-09-29 to the deadline: **31 days (about 4.4 weeks)**.

## Mandatory project and submission requirements

- [ ] Create a working software application that fits one of the four tracks.
- [ ] Make a functional runtime call to the Nebius Token Factory inference API **or** deploy/run the project using Nebius AI Cloud compute (Serverless Jobs, Serverless Endpoints, or DevPods).
- [ ] Use at least one NVIDIA open-source/open model as a functional part of the project. Nemotron is emphasized; GROOT, Cosmos, and Sonic are named for Physical AI.
- [ ] Identify the selected track.
- [ ] Submit a text description of features, functionality, motivation, and operation.
- [ ] Provide a working demo/hosted app/test-build URL. This URL is not required for a Physical AI submission.
- [ ] Provide a public GitHub, GitLab, or Bitbucket repository containing all source code, assets, and instructions needed for the project to function.
- [ ] Include an open-source license file such as Apache 2.0, MIT, or MPL 2.0, visible/detectable at the top of the repository page.
- [ ] Include a README with reproducible setup and run instructions. Highlight the NVIDIA model and every Nebius service actually used; discuss Token Factory only if the project uses it.
- [ ] Upload a public YouTube demo with audio. Keep it **under three minutes** because judges need not watch beyond three minutes.
- [ ] Show the project functioning on its intended device and explain the Nebius and NVIDIA usage in the video.
- [ ] Use only third-party trademarks, music, footage, data, SDKs, and other material that the team is authorized to use.
- [ ] Provide written feedback on the Nebius AI Cloud/Token Factory services actually used and the NVIDIA tools, models, or technologies used.
- [ ] If the project predates 2026-08-26, explain the significant updates made during the submission period.
- [ ] Keep all submission materials in English or provide English translations.
- [ ] Make the project available free of charge and without restriction for judging through the end of the judging period. For uncommon proprietary hardware, be prepared for a possible request for physical access.

Stage One is a pass/fail assessment of baseline viability, genuine track fit, and reasonable use of the required APIs/SDKs. The checklist above combines project and submission requirements; it is not an official itemized definition of the Stage One test.

## Physical AI track

The track covers embodied and edge agents that sense and act in the real world. The official rules say the video should include at least one minute of:

- physical hardware/robot operation; or
- key application modules operating if the project has no physical hardware component.

This project has no accessible physical hardware, so it qualifies for the rules' module-only alternative. Plan for **at least one minute showing the key modules operating together**: a synchronized CREMA-D or live camera/microphone window, the fine-tuned NVIDIA Nemotron visual expert and trained audio expert (trained on Grace), quality-aware fusion, a bounded Nemotron reasoning agent making a real Nebius Token Factory runtime call, deterministic execution validation, NVIDIA Magpie emotional speech, and the resulting official Reachy emotions-library motion in MuJoCo. Label the simulation clearly and do not imply that hardware was tested.

The application will use the supported Reachy Mini SDK and recorded-move interface so the simulator connection can later be replaced by a physical robot connection. Physical motion, latency, audio, stop, and recovery behavior remain unvalidated until a robot is available.

Simulation is compliant with the stated video alternative, but live real-world input should drive the simulated response. A prerecorded motion sequence by itself would be weak evidence that the project senses and acts.

## Judging

Stage Two weights all four criteria equally:

- **Technological Implementation:** build quality and effective use of Nebius plus NVIDIA open models.
- **Design:** a complete, coherent product experience rather than only a technical proof of concept.
- **Potential Impact:** a credible, specific case for a real audience, supported by what is demonstrated.
- **Quality of the Idea:** creative, non-obvious use of the required platforms/models and genuine understanding of the problem.

## Eligibility and ownership notes

- Entrants must be at or above the age of majority where they reside. Teams and eligible organizations are allowed; a team/organization appoints one representative.
- The listed exclusions include residents of Brazil, Quebec, Russia, Crimea, Cuba, Iran, and North Korea, plus sanctioned jurisdictions and sponsor/judge conflicts described in the rules.
- The submission must be the entrant's own work and must not violate intellectual-property, privacy, publicity, or other third-party rights.
- Existing projects are eligible only if significantly updated during the submission period and the update is explained.
- Multiple submissions are allowed only when each is unique and substantially different.

## Credits and optional awards

- The resources page documents $25 in Token Factory credits using activation code `NEBIUS-DEVPOST-GLOBAL26`.
- Joining the Nebius Builders Program documents another $25 in Token Factory credits plus access to other builder resources.
- No general GPU-hour grant for Nebius AI Cloud is documented on the hackathon resources page; verify cost before choosing dedicated infrastructure.
- Best Use of Tavily is a $3,000 bonus and requires a functional runtime Tavily API call. Add it only if it materially improves the product.
- City Winner eligibility is tied to the official rules for the listed participating events; do not assume eligibility without checking the submission form/rules.

## Prizes

Grand Prize $20,000 · 2nd $10,000 · 3rd $6,000 · four track winners (one NVIDIA Jetson Orin Nano each) · Best Use of Tavily $3,000 · 20 City Winners $500 each · 10 Most Valuable Feedback awards of $100 plus NVIDIA swag.

## Repository readiness (current audit: 2026-09-30)

| Requirement | Current status |
|---|---|
| Implementation source | Dataset, perception, harness, integration, sample-media smoke test, and HPRC preflight implemented; live path unverified |
| README with setup/run instructions | Added; must update with actual deployment results |
| License file | Added (MIT project source); third-party terms remain separate |
| Dependency lock/pins | Transformers 4.57.3 and PEFT 0.18.0 pinned for Nemotron compatibility; a complete dependency lock is still missing |
| Nebius Token Factory runtime call evidence | Missing; Grace training alone does not satisfy it |
| VLM one-step fine-tuning feasibility and adapter reload | User-reported Colab A100 baseline step and reload passed; archive its artifacts/logs |
| Audio encoder/head feasibility, license evidence, and reload | Scripted; real model step, reload, and weight-license evidence missing |
| Paired audio-visual fusion feasibility and calibration | Implemented and unit-tested; real paired scores missing |
| Unified audio-visual Nemotron extension | Implemented; user-reported A100 joint optimizer/gradient/ablation/reload pilot passed at 17.19 GiB peak allocated VRAM; processor-fix re-verification and held-out quality pending |
| Visual/audio/fusion held-out evaluation | Evaluation command implemented; real test metrics missing |
| CREMA-D paired manifest, alignment checks, and actor-disjoint split | Implemented and unit-tested; user-reported Colab preparation completed in `cremad-prepared-run2`; archive its validation/provenance reports |
| CREMA-D license and attribution record | Official terms linked in README; prepared output writes source commit and license/metadata hashes; archive the completed Colab run's records |
| Reachy emotions-library move validation | Adapter and allowlist implemented; real library cache/move playback missing |
| Reasoning-agent schema, tools, and scenario evaluation | Schema/guard and offline scenarios tested; live model and broader scenarios missing |
| Magpie emotional voice discovery and TTS evidence | Adapter implemented; live voice list and synthesis missing |
| Guarded agent-to-speech-and-motion integration | Offline mock passes; live simulator/audio synchronization unverified |
| Speech-motion synchronization measurement | Mocked client-dispatch offset implemented; live speaker/motion onset measurement missing |
| Integrated key-module simulation demo | Missing |
| Test/evaluation results | Offline unit tests pass; model metrics and simulator evidence missing |
| Public demo video | Missing |
| Devpost description and feedback | Not yet prepared |

The repository foundations are now present, but code and mocks are not substitutes for measured deployment evidence. The revised training gate in [PROJECT.md](PROJECT.md) keeps visual/audio/fusion results as baselines and additionally requires a real joint audio-visual optimizer step, projector/LoRA gradient checks, modality ablations, GPU profiling, and fresh-process reload before full unified training. A real Nebius Token Factory runtime call, official Reachy emotion move in MuJoCo, and Magpie emotional speech remain separate integration/submission evidence gaps.
