## IV. Reproducible Implementation Specification

This section specifies the system model, implementation contracts, compilation procedure, runtime semantics, reconfiguration protocol, and artifact requirements needed to independently reproduce the proposed typed compile-then-run DAG runtime. The specification intentionally separates semantic requirements from implementation-specific optimizations so that an independent implementation can reproduce the same correctness properties without relying on undocumented behavior of a particular Python interpreter.

### A. Implementation Scope and Execution Assumptions

The reference implementation targets a single video-analytics pipeline executed by one logical pipeline executor on one device. A pipeline processes frames sequentially, with at most one admitted frame executing inside a given pipeline at any instant. Input acquisition may continue through a bounded source queue, but a new frame is admitted to the DAG only after the preceding frame has completed.

The initial implementation supports:
- typed directed acyclic workflows;
- static compilation into immutable execution plans;
- structural runtime reconfiguration through node addition, removal, replacement, and edge rewiring;
- background preparation of candidate plans;
- activation of a candidate plan between two frame executions;
- preservation of unchanged compatible processor instances;
- explicit reset or rejection for unsupported stateful changes;
- failure-atomic candidate preparation.

The following mechanisms are outside the scope of the reference implementation:
- distributed execution;
- simultaneous execution of multiple frames in one pipeline;
- arbitrary state transformation;
- cross-device state migration;
- concurrent reconfiguration transactions;
- online mutation of an already published plan;
- dashboard delivery, database synchronization, and model-artifact transport.

These exclusions ensure that the consistency claim applies to a precisely defined execution regime rather than to an unspecified distributed environment.

### B. Workflow Specification

A workflow is represented as a typed directed graph

$$G=(V,E),$$

where $V$ is a finite set of processor nodes and $E\subseteq V\times V$ is a set of directed data dependencies.

Each node $n\in V$ is defined as

$$n=(id,type,configuration,I,O),$$

where:
- $id$ is a stable identifier unique within the workflow;
- $type$ identifies a registered processor implementation;
- $configuration$ contains processor parameters;
- $I$ is the set of declared input pins;
- $O$ is the set of declared output pins.

Each pin $p$ has:
- a unique local name;
- a payload type $\tau(p)$;
- a cardinality;
- a required or optional status.

An edge is represented as

$$e=(n_s,p_s,n_d,p_d),$$

where $p_s\in O(n_s)$ and $p_d\in I(n_d)$.

A workflow is structurally valid if:
- all node identifiers are unique;
- every node type is registered;
- every referenced source node exists;
- every referenced pin exists;
- every required input has exactly one valid producer, unless the processor explicitly supports multiple producers;
- every edge satisfies the type-assignability relation;
- the graph is acyclic;
- every node configuration satisfies its registered schema.

The type-assignability relation is written as

$$\tau(p_s)\preceq\tau(p_d),$$

meaning that a value produced at $p_s$ may be consumed by $p_d$. The implementation shall define this relation explicitly rather than silently treating unresolved types as unrestricted values.

None shall not be used to represent a missing input because None may be a valid application value. The runtime shall instead use a dedicated singleton sentinel, denoted as MISSING.

### C. Processor Contract

Every processor implementation shall conform to a common interface.

```python
class Processor(Protocol):
    descriptor: ProcessorDescriptor

    def setup(self, context: SetupContext) -> None:
        ...

    def process(self, inputs: object, context: FrameContext) -> object:
        ...

    def healthcheck(self) -> None:
        ...

    def cleanup(self) -> None:
        ...
```

A ProcessorDescriptor shall contain at least:

``` python
@dataclass(frozen=True)
class ProcessorDescriptor:
    type_name: str
    input_schema: type
    output_schema: type
    config_schema: type
    state_policy: StatePolicy
    state_schema_version: str | None
```

The supported state policies are:
- STATELESS: the processor has no cross-frame mutable state;
- PRESERVABLE: the processor state may be reused when its compatibility requirements are satisfied;
- RESETTABLE: a changed processor may be activated with explicitly reset state;
- MIGRATABLE: reserved for future state-transformation support.

The initial paper implementation shall support STATELESS, PRESERVABLE, and explicit RESETTABLE behavior. General MIGRATABLE behavior is excluded.

A processor instance may be reused between plan versions only if:

$$Reusable(n_v,n_{v+1}) = \begin{cases} true,& \text{if all compatibility conditions hold},\\ false,& \text{otherwise}. \end{cases}$$

The compatibility conditions are:
- the stable node identifier is unchanged;
- the processor type is unchanged;
- the state-schema version is unchanged;
- the processor declares that the configuration change preserves state semantics;
- the reconfiguration request does not explicitly require reset.

A stateful node replacement that does not satisfy these conditions shall be rejected unless an explicit reset policy is supplied.

### C.1. Stateful Processor Semantics

A stateful processor maintains mutable information across frame executions. Examples include object trackers, idle-time detectors, temporal aggregators, and event-deduplication processors.

For a tracker, the state may include:
- active track identifiers;
- track histories;
- estimated velocities;
- missed-frame counters;
- temporal confidence;
- object-class associations;
- last-observed frame identifiers and timestamps.

Runtime plan replacement must therefore distinguish structural plan consistency from state continuity. Frame-plan consistency guarantees that one frame is executed using one plan version. It does not, by itself, guarantee that state carried between plan versions remains semantically valid.

#### 1) Stateful processor descriptor

Each stateful processor shall declare:

```python
@dataclass(frozen=True)
class StatefulProcessorDescriptor:
    state_schema_version: str
    supported_transition_policies: frozenset[StateTransitionPolicy]
    preserves_state_for: Callable[
        [ProcessorConfiguration, ProcessorConfiguration, TransitionContext],
        bool,
    ]
```

The supported transition policies are:
- `PRESERVE`
- `RESET`
- `REJECT`

A future implementation may add:

- `MIGRATE`
    but general state transformation is outside the initial scope.

#### 2) Preserve policy

The same processor instance may be referenced by the candidate plan if all preservation conditions hold.

For node $n_v$ in plan $P_v$ and node $n_{v+1}$ in plan $P_{v+1}$,

$$Preservable(n_v,n_{v+1})$$

is true only if:
- the stable node identifier is unchanged;
- the processor type is unchanged;
- the state-schema version is unchanged;
- the processor explicitly declares the configuration transition state-compatible;
- the input payload type is unchanged;
- the semantic coordinate system is unchanged;
- the reconfiguration does not change object-label interpretation in a way that invalidates existing state;
- the request does not explicitly require reset.

For tracking processors, preservation may additionally require:
- unchanged image dimensions or a declared coordinate transformation;
- unchanged bounding-box coordinate convention;
- compatible class-label mapping;
- compatible source identity;
- monotonically increasing frame timestamps;
- unchanged tracker algorithm and state representation.

Changing a downstream visualization node does not normally invalidate tracker state.

Changing the detector, input resolution, label taxonomy, coordinate scaling, or upstream preprocessing may invalidate tracker state even when the tracker node itself is unchanged. Such transitions shall be rejected or explicitly reset unless the tracker declares them compatible.

#### 3) Reset policy

Under RESET, the candidate plan shall contain a newly initialized processor instance with empty state.
The active processor instance shall not be cleared during candidate preparation.
The reset instance is created in staging and becomes visible only when the candidate plan commits.
The runtime shall emit a state-transition event:

```python
@dataclass(frozen=True)
class StateTransitionEvent:
    node_id: str
    old_plan_version: int
    new_plan_version: int
    policy: StateTransitionPolicy
    reason: str
    committed_at_ns: int
```

State reset shall never occur silently.

For a tracker, the first frames after reset may receive new track identifiers. The system shall not claim identifier continuity across an explicit reset.

#### 4) Reject policy

A reconfiguration request shall be rejected if it requires state preservation but the compatibility predicate fails and no explicit reset is allowed.

Examples include:
- replacing a tracker with another implementation;
- changing the track-state schema;
- changing an upstream coordinate representation;
- changing the source stream;
- replacing a detector with an incompatible class taxonomy;
- rewiring a stateful node to semantically different inputs;
- modifying parameters that alter the interpretation of existing state.

Under rejection:
- the active plan remains unchanged;
- the active stateful processor continues processing frames;
- candidate-exclusive resources are cleaned;
- the reason is recorded in the reconfiguration log.

#### 5) Candidate preparation

A preserved processor instance shall not be reinitialized, cleaned, warmed up, or invoked during candidate preparation.
The candidate plan stores a reference to the preserved instance, but the instance remains exclusively owned by the active executor until commit.
A reset processor is prepared as a separate staged instance.
Therefore, candidate preparation must distinguish:
- reused processor reference
    from:
- new staged processor instance

#### 6) Commit boundary

The initial implementation admits at most one frame at a time within a pipeline.
A plan transition occurs only after frame $f_k$ has completed and before frame $f_{k+1}$ is admitted.
For a preserved tracker:
Plaintext

```
frame f_k
    → old plan
    → tracker state S_k

commit P_v → P_{v+1}

frame f_{k+1}
    → new plan
    → same tracker instance initialized with S_k
```

No frame from the old and new plans invokes the tracker concurrently.

This boundary is necessary because preserving a mutable processor instance while multiple frame versions execute concurrently would require additional state-versioning or synchronization semantics.

#### 7) Stateful continuity property

Let

$$S_n(f_i)$$

denote the state of processor $n$ after processing frame $f_i$.

For a preserved processor, state continuity requires:

$$S_n^{v+1}(f_{i+1}) Transition_n \left( S_n^v(f_i), input(f_{i+1}) \right),$$

where Transition is the same declared state transition function used before reconfiguration.

This property is applicable only when the preservation predicate holds.

For an explicit reset:

$$S_n^{v+1}(f_{i+1}) Transition_n \left( S_{n,0}, input(f_{i+1}) \right),$$

where $S_{n,0}$ is the processor's initial state.

#### 8) Removal of a stateful processor

When a stateful node is removed:
- it remains active until the old plan finishes its final frame;
- the new plan is committed;
- the removed processor is retired;
- cleanup occurs only after no active plan references it.

Any final state snapshot required for audit shall be obtained before cleanup.
#### 9) Stateful source and sink processors

Long-lived source and sink services require separate treatment.
A video-source service may preserve:
- decoder connection;
- source timing state;
- frame counter;
- reconnect state.

An RTMP sink may preserve:
- encoder session;
- RTMP connection;
- presentation timestamp generator.

These services should normally remain outside the replaceable analytics-plan state and be referenced by plan nodes through stable service handles. This prevents internal graph edits from interrupting source decoding or output streaming.

#### 10) Tracking-specific validation tests

The artifact shall include the following tests.

**Preservation test**
- execute a tracker until stable track identifiers exist;
- add or remove a compatible stateless downstream node;
- commit the new plan;
- verify that the tracker instance identity is unchanged;
- verify that track identifiers and histories continue.

**Reset test**
- execute a tracker until state exists;
- submit a reconfiguration with explicit RESET;
- commit the candidate;
- verify that a new tracker instance is used;
- verify that the reset event is recorded.

**Rejection test**
- execute a tracker until state exists;
- submit an incompatible upstream or tracker configuration change;
- request preservation;
- verify that the candidate is rejected;
- verify that the active tracker state remains unchanged.

**No-concurrent-access test**

Instrument processor entry and exit. Verify that the preserved processor is never invoked concurrently by old and new plans.

**State-leak test**

Repeatedly alternate reset and removal operations. Verify that retired tracker instances are eventually cleaned and do not accumulate in memory.

#### 11) Stateful measurements

The evaluation shall report:

- preserved-state transition count;
    
- explicit-reset count;
    
- rejected-state transition count;
    
- tracker-instance reuse count;
    
- state continuity failures;
    
- duplicate or discontinuous track identifiers;
    
- stateful-node commit latency;
    
- retirement latency;
    
- leaked processor-instance count.
    

The paper shall distinguish:

- frame-plan consistency;
    
- state continuity for declared-compatible transitions;
    
- explicit state reset;
    
- unsupported general state migration.

### D. Registry Snapshot

Processor resolution shall be performed through a registry

$$\mathcal{R}: \text{type\_name}\rightarrow \text{ProcessorDescriptor}.$$

Compilation shall use an immutable registry snapshot. This prevents a processor registration change from altering the meaning of a candidate plan while it is being compiled.

The registry snapshot identifier shall be stored in the resulting execution plan. Consequently, the same normalized workflow specification, compiler version, and registry snapshot shall produce the same structural plan representation.

### E. Compiler Architecture

Compilation shall be implemented as a separate, side-effect-controlled component:

```python
class WorkflowCompiler:
    def compile(
        self,
        specification: WorkflowSpecification,
        registry: RegistrySnapshot,
        previous_plan: ExecutionPlan | None = None,
        state_directive: StateDirective | None = None,
    ) -> CandidatePlan:
        ...
```

The compiler shall not modify the currently active execution plan.

Compilation consists of the following phases.

**1) Parsing and normalization**

The input JSON document is parsed into immutable schema objects. Equivalent syntactic representations shall be normalized into one canonical representation. Node ordering in the source document shall not determine runtime execution order.

The normalized specification shall be serialized canonically and hashed:

$$h_G=H(\operatorname{Canonicalize}(G)).$$

The hash shall be stored as part of the plan provenance record.

**2) Structural validation**

The compiler validates:
- duplicate node identifiers;
- unknown processor types;
- malformed edge references;
- unknown input and output pins;
- missing required inputs;
- unsupported multiple producers;
- invalid node configurations.

Any validation failure aborts compilation.

**3) Type validation**
For every edge $e$, the compiler evaluates

$$\tau(p_s)\preceq\tau(p_d).$$

Generic pins shall be resolved before processor initialization. A generic group that resolves to incompatible concrete types shall produce a deterministic compilation error.

**4) Cycle detection and topological ordering**

The initial implementation shall reconstruct the complete ordering using Kahn's algorithm.

Let $d^-(v)$ denote the in-degree of node $v$. Compilation begins with all nodes satisfying $d^-(v)=0$, repeatedly removes one node, and decreases the in-degree of its dependents. Compilation succeeds only if the resulting order contains all nodes:

$$\vert{}O\vert{}=\vert{}V\vert{}.$$

Otherwise, the workflow contains a cycle and is rejected.

Incremental topological sorting is not required for the initial implementation. Full reconstruction is performed outside the active execution path and is separately benchmarked as a function of graph size.

**5) Processor-difference analysis**

When a previous plan is supplied, the compiler classifies each node as:
- unchanged and reusable;
- configuration-updated but reusable;
- added;
- removed;
- replaced;
- rewired.

This classification determines which processor instances can be referenced by the candidate and which instances must be prepared in staging.

**6) Execution-step construction**

Each compiled step shall contain only data needed by the frame-execution loop:
```python
@dataclass(frozen=True, slots=True)
class ExecutionStep:
    node_id: str
    output_index: int
    processor_ref: Processor
    input_bindings: tuple[InputBinding, ...]
    readiness_rule: ReadinessRule
    output_schema: type
```

An input binding is defined as:

```python
@dataclass(frozen=True, slots=True)
class InputBinding:
    destination_pin: str
    source_index: int
    source_pin: str
    required: bool
```

String node identifiers are resolved into integer output-slot indices during compilation.

The reference implementation shall define input-construction semantics independently from any CPython code-generation optimization. If specialized functions generated through compile or exec are used, this optimization shall be isolated behind an InputFactory interface and evaluated through an ablation study.

**7) Conditional execution**

Conditional execution shall not be implemented by unconditionally skipping every transitive descendant of a gate. Such behavior may incorrectly suppress a merge node that remains executable through another branch.

Instead, each execution step shall use an explicit readiness rule. A node executes only if:
- every required input is available;
- its branch predicate, when present, evaluates to true;
- its processor-specific activation condition is satisfied.

This rule permits safe fork, merge, optional-input, and bypass structures.

### F. Immutable Execution Plan

Successful compilation produces a candidate plan:

$$P_v=(v,O,S,R,M,Q),$$

where:
- $v$ is a monotonically increasing version;
- $O$ is the topological order;
- $S$ is the immutable tuple of compiled execution steps;
- $R$ is the processor-lifecycle record;
- $M$ is plan metadata and provenance;
- $Q$ is the required frame-workspace specification.

A suggested Python representation is:

```python
@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    version: int
    specification_hash: str
    registry_snapshot_id: str
    compiler_version: str
    steps: tuple[ExecutionStep, ...]
    output_slot_count: int
    processor_ids: frozenset[str]
```

Published structural metadata shall never be modified in place.

Processor objects referenced by a plan may contain state, but their identity and structural binding to the plan shall remain fixed. Because the reference executor admits one frame at a time and commits only between frames, preserved stateful instances are never concurrently invoked by two plan versions.

### G. Static Frame Execution

For each admitted frame, the executor obtains one local reference to the current plan and uses that reference for the complete frame.

**Algorithm 1: Execute one frame**

Input:

frame $f$

current execution plan $P$
1. plan ← $P$
2. workspace ← acquire_workspace(plan)
3. clear workspace using MISSING
4. create FrameContext(frame_id, plan.version, timestamps)
5. for step in plan.steps do
    
6. ```
    if readiness_rule(step, workspace) is false then
    ```
    
7. ```
        workspace[step.output_index] ← MISSING
    ```
    
8. ```
        continue
    ```
    
9. ```
    inputs ← construct_inputs(step.input_bindings, workspace)
    ```
    
10. output ← step.processor_ref.process(inputs, FrameContext)
11. workspace[step.output_index] ← output
12. emit configured sink outputs
13. release_workspace(workspace)
14. return FrameResult(frame_id, plan.version)

The executor shall not read a global active-plan reference inside the node loop.

The workspace is plan-specific because different plan versions may contain different numbers of output slots. A workspace pool may be used as an optimization, but reuse shall not expose values from a preceding frame.

### G.1. Decoupled Streaming and Inference Cadences

A video source may produce frames at a substantially higher rate than the inference pipeline can process. Let $F_s$ denote the source-frame rate and $F_i$ denote the effective inference rate. In practical deployments, it is common that

$$F_i < F_s,$$

for example, a 30-frame/s camera processed by a detector operating at 5 inference cycles/s.

The output-stream cadence shall therefore not be controlled by the completion of inference cycles. If the encoder waits for every inference result before publishing a frame, inference variability directly produces output jitter, increased end-to-end latency, and potentially unstable RTMP sessions.

The reference implementation shall separate:
- source-frame acquisition;
- frame selection for inference;
- annotation publication;
- presentation-frame rendering;
- encoding and RTMP transmission.

The resulting architecture is:

Plaintext

```
Video Source
    │
    ├── every source frame ──► Presentation Queue ──► Renderer ──► Encoder/RTMP
    │                                                   ▲
    │                                                   │
    └── latest-frame slot ───► DAG Inference ──► Annotation Slot
```

#### 1) Frame envelope

Every decoded frame shall be represented by an immutable envelope:

```python
@dataclass(frozen=True, slots=True)
class FrameEnvelope:
    source_id: str
    frame_id: int
    pts_ns: int
    captured_at_ns: int
    image: ReadOnlyFrameBuffer
    width: int
    height: int
```

The source presentation timestamp, `pts_ns`, shall remain the authoritative time reference for output pacing. Inference-completion timestamps shall not replace the source timestamp.

Captured frame buffers shall be treated as read-only. The inference executor may read the original buffer, whereas the renderer shall obtain a private writable buffer or a buffer from a rendering pool before drawing overlays.

#### 2) Latest-frame inference admission

The inference path shall not use an unbounded FIFO queue. Otherwise, when $F_i < F_s$, the inference executor increasingly processes stale frames.

Instead, the capture service shall publish frames into a single-slot latest-value channel:

```python
class LatestFrameSlot:
    def publish(self, frame: FrameEnvelope) -> None:
        ...

    def take_latest(self) -> FrameEnvelope | None:
        ...
```

Publishing a newer frame replaces an older frame that has not yet been admitted to inference. Frames replaced in this channel are classified as intentionally skipped analysis frames. They are not considered streaming failures because they may still be rendered and transmitted at the source cadence.

The artifact shall report separately:
- captured frames;
- inference-admitted frames;
- intentionally skipped analysis frames;
- rendered frames;
- encoder-dropped frames;
- transmitted frames.

#### 3) Annotation snapshot

Dynamic overlay information shall be represented as immutable data rather than as a newly allocated Python callback that captures the latest detection result.

```python
@dataclass(frozen=True, slots=True)
class AnnotationSnapshot:
    source_id: str
    analysis_frame_id: int
    analysis_pts_ns: int
    inference_completed_at_ns: int
    plan_version: int
    producer_node_id: str
    detections: tuple[ImmutableDetection, ...]
    style: OverlayStyle
```

The drawing implementation remains static:

```python
class OverlayRenderer:
    def render(
        self,
        frame: FrameEnvelope,
        snapshot: AnnotationSnapshot,
    ) -> RenderedFrame:
        ...
```

This separation makes annotation data:
- timestampable;
- versioned;
- testable;
- serializable;
- independent of processor lifetime;
- safe to retain after an inference cycle completes.

#### 4) Latest-annotation publication

Inference results shall be published through a latest-value channel rather than a FIFO queue:

```python
class LatestValueSlot(Generic[T]):
    def publish(self, value: T) -> None:
        ...

    def load(self) -> T | None:
        ...

    def clear(self) -> None:
        ...
```

When a new annotation becomes available, it replaces the previous annotation. Intermediate results that have already been superseded are not rendered later.

Synchronization shall protect only loading, replacing, or clearing the shared reference. Rendering, frame copying, encoding, and network transmission shall occur after the synchronization primitive has been released.

Conceptually:

```python
snapshot = annotation_slot.load()

if snapshot is not None:
    rendered = renderer.render(frame, snapshot)
```

The implementation shall not hold the annotation lock while executing render, encoder calls, or RTMP writes.

#### 5) Annotation-selection policy

For a presentation frame $F_j$ with timestamp $t_j$, let $A_k$ denote an annotation produced from an analysis frame with timestamp $a_k$.

The renderer selects the most recently published annotation satisfying:

$$a_k \le t_j.$$

Among eligible annotations, it chooses:

$$k^*(j)=\max\{k\mid a_k\le t_j\}.$$

The annotation is applied only if its temporal age remains within a configured bound:

$$0\le t_j-a_{k^*(j)}\le\Delta_{\max},$$

where $\Delta_{\max}$ is the maximum permitted annotation age.

If the latest annotation exceeds this bound, the renderer shall emit the presentation frame without that annotation rather than displaying an indefinitely stale bounding box.

The initial implementation shall provide at least the following policy:

**LATEST_WITH_TTL**

Under this policy, the latest valid annotation is reused until:
- a newer annotation is published;
- its time-to-live expires;
- its source identifier no longer matches;
- its plan version is no longer accepted.

This policy maintains a source-driven output cadence but does not claim exact frame-to-detection alignment.

#### 6) Temporal semantics

The implementation shall explicitly distinguish:
- video continuity;
- annotation freshness;
- exact temporal alignment.

Reusing the latest annotation preserves output continuity, but an annotation generated from one frame may be displayed on later frames. Therefore, the mechanism provides bounded annotation staleness rather than exact frame correspondence.

Exact correspondence would require either:
- delaying presentation frames until matching inference results become available; or
- predicting object motion between detector executions.

Both mechanisms introduce additional latency or algorithmic assumptions and are outside the initial runtime contribution.

#### 7) Presentation queue

The capture-to-render path shall use a bounded queue. If the renderer or encoder temporarily falls behind, the queue shall apply an explicitly documented policy.

For low-latency monitoring, the recommended policy is:

**drop oldest presentation frame**

rather than allowing queue growth and increasing end-to-end latency.

The queue capacity and drop policy shall be reported in every experiment.

#### 8) Presentation-frame consistency

Let $Draw(F_j)$ denote all drawing operations applied to presentation frame $F_j$. Let $R_q$ denote an immutable render state containing the accepted plan version and current annotation snapshot.

The renderer is presentation-frame consistent if:

$$\forall F_j,\exists!q: \forall d\in Draw(F_j),\operatorname{renderState}(d)=R_q.$$

The renderer shall load the render state once before drawing a frame and shall not reload shared annotation state while processing that frame.

Python

```
render_state = render_state_slot.load()
rendered_frame = render_frame(frame, render_state)
```

A concurrently published annotation may therefore affect the next presentation frame, but not alter a frame whose rendering has already begun.

#### 9) Interaction with plan reconfiguration

Annotation state shall be associated with an execution-plan version.

A combined immutable render state is recommended:

```python
@dataclass(frozen=True, slots=True)
class RenderState:
    accepted_plan_version: int
    annotation: AnnotationSnapshot | None
```

Plan version and annotation shall not be published through unrelated mutable variables because the renderer could otherwise observe a new plan version together with an annotation produced by the preceding plan.

When plan $P_v$ is replaced by $P_{v+1}$, the reconfiguration controller shall publish:

```python
RenderState(
    accepted_plan_version=v + 1,
    annotation=None,
)
```

before the next frame is rendered under the new analytics semantics.

The stream shall continue without an overlay until the first annotation produced by $P_{v+1}$ becomes available.

An annotation shall be rejected when:

$$annotation.plan_version \ne renderState.accepted_plan_version.$$

This prevents an annotation produced by an old graph version from being drawn after a structural reconfiguration has committed.

#### 10) Persistent RTMP service

The RTMP encoder and publisher should be implemented as a persistent runtime service rather than as a processor that is destroyed whenever the analytics DAG changes.

The logical workflow may contain an RTMP output node, but its implementation shall reference a stable streaming service when the following properties remain unchanged:
- endpoint;
- codec;
- output resolution;
- frame rate;
- time base;
- encoder configuration.

A structural modification to the analytics subgraph shall therefore not reconnect the RTMP session.

A change to the endpoint, codec, time base, or resolution is classified as sink reconfiguration and may require encoder restart or reconnection. Such changes shall be evaluated separately from internal DAG reconfiguration.

#### 11) Streaming measurements

The evaluation shall report:
- source frame rate;
- inference frame rate;
- output RTMP frame rate;
- output inter-frame interval;
- inter-frame jitter;
- maximum output gap;
- annotation-age median;
- annotation-age p95 and p99;
- stale-annotation suppression count;
- presentation-frame drop count;
- analysis-frame skip count;
- plan-version mismatch count;
- RTMP reconnection count.

A smooth output stream shall not be inferred solely from average FPS. Inter-frame interval distributions and maximum gaps shall also be reported.

### H. Runtime Reconfiguration Protocol

A reconfiguration request is defined as:

```python
@dataclass(frozen=True)
class ReconfigurationRequest:
    request_id: str
    base_version: int
    target_specification: WorkflowSpecification
    state_directive: StateDirective
    submitted_at_ns: int
```

The base_version prevents lost updates. A candidate prepared from version $v$ shall not be committed if the active version has already advanced beyond $v$.

Each request proceeds through the state machine:

RECEIVED
↓
VALIDATING
↓
PREPARING
↓
READY
↓
COMMITTED

A request may instead terminate as:

REJECTED, FAILED, STALE, ABORTED

**1) Off-path candidate compilation**

A compiler thread or worker pool receives the request and constructs a candidate without pausing the active executor.

**2) Staged resource preparation**

Processors that cannot be reused are created in a staging area. For each new processor, the runtime performs:
- construction;
- configuration validation;
- setup;
- resource allocation;
- optional model loading;
- optional warm-up;
- healthcheck.

A reused processor is referenced but is not reinitialized during candidate preparation.

**3) Candidate readiness**

A candidate becomes READY only if every required processor and resource has been prepared successfully.

**4) Boundary commit**

The executor checks for a ready candidate only after the current frame has completed and before admitting the next frame.

**Algorithm 2: Commit a candidate plan**
1. finish execution of frame $f_k$ under $P_v$
2. obtain next READY candidate $C$
3. if $C$.base_version ≠ active_plan.version then
4. ```
    mark $C$ as STALE
    ```
    
5. ```
    clean staged resources
    ```
    
6. ```
    return
    ```
7. old_plan ← active_plan
8. active_plan ← $C$.plan
9. record commit timestamp
10. mark $C$ as COMMITTED
11. retire resources in old_plan that are not reused
12. admit frame $f_{k+1}$ under the new active plan

The plan-switch assignment is the linearization point of the reconfiguration.

A thread-safe queue is sufficient to transfer a ready candidate from the compiler to the single executor. The active plan itself does not require mutation by the compiler thread.

### I. Frame-Plan Consistency

**Definition 1: Frame-plan consistency**

Let $E(f)$ be the set of all execution events associated with frame $f$, including input construction, readiness evaluation, processor invocation, routing, and output emission.

The runtime is frame-plan consistent if:

$$\forall f,\exists!v: \forall e\in E(f),\operatorname{version}(e)=v.$$

Thus, exactly one execution-plan version determines the complete processing of each admitted frame.

**Proposition 1**

Assume that:
- published plan metadata is immutable;
- at most one frame is executing in a pipeline;
- plan activation occurs only after one frame has completed and before the next frame is admitted;
- the executor uses one local plan reference throughout frame execution;
- reconfiguration requests are serialized.

Then every admitted frame is processed under exactly one execution-plan version.

**Proof sketch**

Consider the sequence of admitted frames

$$f_1,f_2,\ldots,f_k.$$

For the base case, $f_1$ is admitted after the executor selects one active plan $P_v$. The executor does not change or reread the active-plan reference during the frame, so every event associated with $f_1$ uses $P_v$.

Assume that every frame up to $f_i$ is processed under exactly one plan. After $f_i$ completes, the executor may either retain the current plan or commit one ready candidate. The commit occurs before $f_{i+1}$ is admitted. Therefore, $f_{i+1}$ observes either the previous plan or the committed candidate, but not both. Because no subsequent plan lookup occurs during its execution, all events of $f_{i+1}$ use the selected plan. By induction, the property holds for every admitted frame.

The proposition establishes a safety property. It does not guarantee that every submitted candidate will eventually compile or commit.

### J. Failure Atomicity

**Proposition 2**

If candidate compilation, validation, setup, warm-up, or health checking fails before boundary commit, the active execution plan and its processor state remain unchanged.

This property follows from three design rules:
- the compiler never mutates the active plan;
- new resources are prepared in staging;
- publication occurs only after complete candidate preparation.

On failure, all candidate-exclusive resources shall be cleaned, and the request shall terminate as REJECTED or FAILED.

Failure injection shall cover at least:
- cyclic graph;
- incompatible pin type;
- missing required input;
- unknown processor type;
- invalid processor configuration;
- processor-constructor exception;
- setup exception;
- model-loading failure;
- health-check failure.

### K. Event Logging and Measurement

Every reproducible execution shall produce an append-only event log.

For each frame:
- frame_id
- plan_version
- admission_timestamp_ns
- completion_timestamp_ns
- executed_node_ids
- skipped_node_ids
- status

For each reconfiguration:
- request_id
- base_version
- candidate_version
- specification_hash
- request_received_ns
- validation_completed_ns
- preparation_completed_ns
- ready_ns
- commit_ns
- first_new_frame_admitted_ns
- first_new_frame_completed_ns
- retirement_completed_ns
- terminal_status
- failure_reason

The following intervals can then be reconstructed:

$$T_{\text{validation}}$$

$$T_{\text{preparation}}$$

$$T_{\text{request-to-ready}}$$

$$T_{\text{boundary-wait}}$$

$$T_{\text{commit}}$$

$$T_{\text{request-to-effect}}$$

$$T_{\text{retirement}}.$$

Service-continuity measurements shall include:
- maximum output gap;
- dropped-frame count;
- duplicated-frame count;
- mixed-version-frame count;
- queue-depth maximum;
- latency during the reconfiguration interval;
- steady-state latency before and after commit.

### L. Reference Baselines

The artifact shall implement the following baselines using the same processor functions and input trace.
- Hard-coded pipeline: direct processor calls without generalized DAG orchestration.
- Static compiled DAG: compiled execution plan without runtime versioning.
- Versioned compiled DAG: proposed executor with candidate preparation and boundary commit.
- Stop–rebuild–restart: stop execution, destroy the old runtime, compile the new graph, and restart.
- Pause–compile–resume: pause frame admission, compile synchronously, replace the plan, and resume.

Processor logic, model instance, preprocessing, input frames, and sink behavior shall remain identical across baselines.

### M. Verification Test Suite

The reproducibility artifact shall contain four test levels.

**1) Compiler unit tests**

These tests cover:
- valid linear graph;
- valid branch and merge graph;
- duplicate node identifier;
- unknown processor;
- unknown pin;
- missing required input;
- incompatible payload type;
- generic-type conflict;
- cycle detection;
- deterministic topological output;
- deterministic canonical specification hash.

**2) Differential execution tests**

A graph shall be executed by:
- a simple interpreted reference evaluator;
- the compiled executor.

Both implementations shall produce identical outputs and node-invocation sequences.

**3) Reconfiguration integration tests**

These tests cover:
- add node;
- remove node;
- add branch;
- remove branch;
- rewire edge;
- replace stateless processor;
- preserve compatible stateful processor;
- explicit state reset;
- stale candidate rejection.

**4) Consistency stress tests**

A deterministic sequence of frames and serialized graph updates shall be executed while recording every frame and node version. The checker shall verify:

$$\left\vert{}\{\operatorname{version}(e):e\in E(f)\}\right\vert{}=1$$

for every frame $f$.

The test shall also detect:
- missing frame identifiers;
- duplicate frame identifiers;
- execution through a retired plan;
- output from an undeclared node;
- unresolved required input;
- leaked candidate processors.

Property-based generation may be used to produce random valid and invalid DAGs, provided that the random seed is stored with every result.

### N. Repository Organization

A recommended repository structure is:

```
repository/
├── pyproject.toml
├── requirements.lock
├── README.md
├── Dockerfile
├── src/
│   └── dag_runtime/
│       ├── specification.py
│       ├── type_system.py
│       ├── registry.py
│       ├── processor.py
│       ├── validation.py
│       ├── compiler.py
│       ├── plan.py
│       ├── executor.py
│       ├── reconfiguration.py
│       ├── lifecycle.py
│       ├── workspace.py
│       └── instrumentation.py
├── tests/
│   ├── unit/
│   ├── differential/
│   ├── integration/
│   ├── failure_injection/
│   └── stress/
├── benchmarks/
│   ├── hard_coded/
│   ├── static_compiled/
│   ├── versioned_compiled/
│   ├── restart/
│   └── synchronous_rebuild/
├── workflows/
│   ├── representative/
│   ├── synthetic/
│   ├── invalid/
│   └── reconfiguration_sequences/
├── datasets/
│   └── manifests/
├── scripts/
│   ├── run_tests.py
│   ├── run_overhead_benchmark.py
│   ├── run_reconfiguration_benchmark.py
│   ├── run_consistency_stress.py
│   ├── run_failure_injection.py
│   └── generate_paper_results.py
└── results/
    ├── raw/
    ├── processed/
    ├── figures/
    └── tables/
```

### O. Environment Reproduction

The artifact shall report:
- operating-system version;
- Python implementation and version;
- compiler/runtime commit hash;
- dependency lockfile hash;
- CPU model;
- GPU model and driver;
- CUDA and inference-framework versions;
- available RAM and VRAM;
- storage type;
- process affinity;
- power and clock configuration;
- relevant environment variables.

A container image shall be provided for software dependencies. Hardware-specific libraries that cannot be included shall be documented through a machine-readable environment manifest.

Every benchmark invocation shall store:
- repository commit;
- workflow hash;
- model-artifact hash;
- video-input hash;
- random seed;
- warm-up count;
- measured-frame count;
- repetition count;
- baseline name;
- timestamp;
- hardware identifier.

### P. Reproduction Procedure

An independent reproducer shall be able to perform the following sequence.
1. Clone the artifact repository.
2. Check out the reported commit.
3. Build the container or create the locked Python environment.
4. Verify model and video hashes.
5. Run compiler unit tests.
6. Run differential execution tests.
7. Run reconfiguration integration tests.
8. Run consistency and failure-atomicity tests.
9. Execute the steady-state overhead benchmarks.
10. Execute the reconfiguration benchmarks.
11. Generate processed CSV files.
12. Regenerate every table and figure used in the paper.

The paper shall not require manual transcription of benchmark values. All reported tables and figures shall be generated from committed scripts and raw result files.

### Q. Clean-Room Rewrite Order

The implementation should be developed in the following order.
- Immutable workflow schemas and canonical serialization.
- Processor interface and immutable registry snapshot.
- Structural and typed compiler.
- Static immutable execution plan.
- Sequential static executor.
- Differential tests against an interpreted evaluator.
- Candidate-plan builder and processor staging.
- Frame-boundary reconfiguration controller.
- State-reuse, reset, and rejection policies.
- Reconfiguration event logging.
- Failure injection.
- Benchmark baselines and result-generation scripts.

Dashboard integration, database persistence, external message brokers, model distribution, and threaded processor execution should be added only after the core runtime and its consistency properties have been validated.

### R. Reproducibility Statement

The complete artifact shall include the runtime source code, immutable workflow specifications, processor implementations used in evaluation, invalid-workflow corpus, structural-edit sequences, benchmark scripts, raw timing traces, environment manifests, and scripts that regenerate all reported results. The artifact shall distinguish representative video-analytics workloads from synthetic graph-scaling workloads. Synthetic workloads shall be labeled as stress tests and shall not be presented as representative production pipelines.

Under the stated single-executor and one-frame-in-flight assumptions, an independent implementation conforming to this specification should reproduce the same structural-validation behavior, frame-boundary activation semantics, frame-plan consistency property, and failure-atomic candidate preparation, even if lower-level implementation details differ.