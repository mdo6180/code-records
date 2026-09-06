The **Ziegler–Nichols tuning method** is an empirical way to find reasonable starting values for PID gains without needing an accurate mathematical model of your system.

Given the controllers you've been building, the most relevant version is the **closed-loop / ultimate-gain method**. The basic idea is:

> Increase \(K_p\) until the system is right on the edge of instability, measure how it oscillates, and use those measurements to calculate initial PID gains.

## 1. Start with only proportional control

Set:

$$
K_i = 0,\qquad K_d = 0
$$

so your controller is simply

$$
u(t)=K_p e(t)
$$

For example, for a drone rate controller:

$$
\tau_x = K_p(p_d-p)
$$

Start with a small \(K_p\).

---

## 2. Gradually increase \(K_p\)

Run the system and keep increasing \(K_p\).

You'll typically see something like:

**Small \(K_p\):**

```text
setpoint ────────────────────────

response    _________------------
```

Slow and stable.

**Higher \(K_p\):**

```text
setpoint ────────────────────────

response      /\__
             /    \____----------
```

Faster, perhaps some overshoot.

**Even higher \(K_p\):**

```text
setpoint ────────────────────────

response      /\    /\
             /  \__/  \____
```

Oscillatory, but the oscillations eventually decay.

Eventually you find a special value where the oscillations **neither grow nor decay**:

```text
setpoint ─────────────────────────

response      /\      /\      /\
             /  \    /  \    /  \
            /    \__/    \__/    \__
```

That \(K_p\) is called the **ultimate gain**:

$$
\boxed{K_u}
$$

---

## 3. Measure the oscillation period

Now measure the time between successive peaks.

```text
              peak            peak
                ↓               ↓
                /\              /\
               /  \            /  \
──────────────/────\──────────/────\────
                <------------>
                     T_u
```

This is the **ultimate period**:

$$
\boxed{T_u}
$$

So you now have two experimentally measured numbers:

$$
K_u = \text{gain producing sustained oscillation}
$$

$$
T_u = \text{period of those oscillations}
$$

Suppose your drone simulation produced:

$$
K_u=8
$$

and

$$
T_u=0.4\text{ s}
$$

---

## 4. Calculate your controller gains

The classic Ziegler–Nichols rules are:

| Controller |     \(K_p\) |         \(K_i\) |         \(K_d\) |
| ---------- | ----------: | --------------: | --------------: |
| P          |  \(0.5K_u\) |               — |               — |
| PI         | \(0.45K_u\) | \(0.54K_u/T_u\) |               — |
| PID        |  \(0.6K_u\) |  \(1.2K_u/T_u\) | \(0.075K_uT_u\) |

For our example:

$$
K_u=8,\qquad T_u=0.4
$$

For a PID controller:

$$
K_p=0.6(8)=4.8
$$

$$
K_i=\frac{1.2(8)}{0.4}=24
$$

$$
K_d=0.075(8)(0.4)=0.24
$$

So:

$$
\boxed{K_p=4.8,\quad K_i=24,\quad K_d=0.24}
$$

These would be your **starting gains**, not necessarily your final gains.

---

## Why does this work?

The clever part is finding \(K_u\).

When you increase \(K_p\) until you get sustained oscillation, you've experimentally discovered something about the dynamics of your plant.

You're effectively asking:

> "How much feedback gain can this physical system tolerate before the feedback loop becomes unstable?"

And \(T_u\) tells you:

> "At what characteristic timescale does the system oscillate when it's at that stability boundary?"

Ziegler and Nichols experimentally developed rules for backing away from that instability boundary to produce a reasonably responsive controller.

So instead of deriving an exact model like

$$
I_{xx}\dot p = \tau_x
$$

and doing a full stability analysis, you're letting the **physical system reveal its dynamics experimentally**.

---

## Applying it to your drone rate controller

Suppose you have:

$$
p_d \rightarrow \boxed{\text{roll rate controller}}
\rightarrow \tau_x
\rightarrow \boxed{\text{drone dynamics}}
\rightarrow p
$$

Your current PI controller conceptually looks like:

$$
\tau_x =
K_p(p_d-p)
+
K_i\int(p_d-p)\,dt
$$

To use Ziegler–Nichols, you'd temporarily disable integral action:

```python
kp_p = ...
ki_p = 0.0
```

Then command something like

$$
p_d = 1\text{ rad/s}
$$

and gradually increase `kp_p`.

Suppose:

```text
Kp = 2    stable, slow
Kp = 4    stable, faster
Kp = 6    oscillations decay
Kp = 7    sustained oscillation
Kp = 8    oscillations grow
```

Then you'd estimate:

$$
K_u \approx 7
$$

Measure the period of the oscillation, perhaps:

$$
T_u=0.3\text{ s}
$$

Then, if you want PI:

$$
K_p=0.45(7)=3.15
$$

and

$$
K_i=\frac{0.54(7)}{0.3}=12.6
$$

So you'd start testing around:

```python
kp_p = 3.15
ki_p = 12.6
```

and tune from there.

## One important limitation for your drone

Ziegler–Nichols tends to produce **aggressive gains**. It was designed to give a fast response with significant overshoot rather than the smoothest possible response.

For a drone, I'd treat it as a systematic way of getting into the right neighborhood rather than saying:

> "Ziegler–Nichols says \(K_p=3.15\), therefore 3.15 is correct."

You'd still look at your response and adjust based on rise time, overshoot, settling time, actuator saturation, noise, etc.

And there's an additional issue with doing the classic experiment on a **real drone**: intentionally pushing a flight-control loop to the edge of instability can be dangerous. Your simulator is actually an excellent place to experiment with Ziegler–Nichols.

It also fits nicely with the tuning principle you've been developing:

$$
\boxed{
\text{P only}
\rightarrow
\text{find system behavior}
\rightarrow
\text{add I/D if necessary}
\rightarrow
\text{fine tune}
}
$$

Ziegler–Nichols essentially turns the first part of that process into a repeatable experiment.
