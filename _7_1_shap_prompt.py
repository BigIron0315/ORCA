def get_symbolic_form_prompt(target_KPM):
    prompt = f"""
You are an expert in wireless networks and Open RAN.
You are given a target Key Performance Metric (KPM), denoted as c.
---
**Step 1**: Check whether the KPM {target_KPM} can be represented using one of the following abstract formulations:
For example, Shannon's theorem for capacity, Little's law for delay, and so on.
   - {target_KPM} = max(a, b)
   - {target_KPM} = min(a, b)
   - {target_KPM} = a / b
   - {target_KPM} = a + b
   - {target_KPM} = a - b
   - {target_KPM} = a * b

Use only high-level concepts here, such as:
   - SystemCapacity
   - TrafficDemand
   - BufferStatus
   - SignalQuality
   - Throughput_Mbps
These are examples of high-level concepts.

DL_Buffer : total amount of traffic demand of the gNB
Avg_SNR_dB : average SNR across users in the gNB
Throughput_Mbps : sum throughput of all users in the gNB (approximation of system capacity)
Avg_Delay_ms : average packet end-to-end delay (queuing delay + transmission delay) across users in the gNB
user_throughput : average throughput per user in the gNB
Users : number of users in the gNB
---
**Output Format**:
Line 1: If yes, just abstract equation? If no, just say "NO"
"""

    return prompt

def get_math_equation_prompt(symbolic_form):
    prompt = f"""
Mathematical equation : {symbolic_form}

**Step 2**: Check whether abstract formulation in step 1 can be approximated with measurable parameters.
For example, trafficDemand ≈ DL_Buffer.
Reformulate equation with measurable parameters. You don't need to use all of them, just use parameters that you need.

[Measurable parameters]:
- Avg_SNR_dB:
- ant_tilt_deg:
- CIO:
- TxPower:
- PRB_num:
- TTT:
- HYS:
- Mobility_Score:
- Throughput_Mbps:
- RLF_events:
- Balance_Ratio:
- DL_Buffer : total amount of traffic demand of the gNB

---
**Output Format**:
Line 1: Reformulated equation"""
    return prompt


#
    if "min(" in symbolic_equation:
        prompt = f"""

Please follow these steps in your response:
1. First, explain the original mathematical formulation based on probability:
    - X (=Importance of A) is proportional to [P(A < B)/P(B < A)].
    - Y (=Importance of B) is proportional to [P(B < A)/P(A < B)].
    - In general, it would involve the CDF of (A-B) evaluated at 0, i.e., P(A < B) = F_D(0).
2. Under the assumption that the distribution of A and B are same, approximate this formulation to obtain a closed-form expression without using CDF:
    - Prefer using a ratio-based approximation between A and B, based on their expected values.
    - Based on your approximation find the importance shift ratio β for A and B.
      β_A = 
      β_B = 

Suppose in the original environment:
E[A] = a, E[B] = b
Importance: X for A, Y for B.

New environment
E[A] = a', E[B] = b'
Importance: X' for A, Y' for B.

I need a importance shift factor β_A = X'/X and β_B = Y'/Y

Please follow these steps in your response:
1. First, explain the original mathematical formulation based on probability:
    - C = 
    - X = 
    - Y = 
    - In general, it would involve the CDF of (A−B) evaluated at 0, i.e., P(A < B) = F_D(0).
2. Under the assumption that the distribution of A and B are same, approximate this formulation to obtain a closed-form expression without using CDF:
    - Prefer using a ratio-based approximation between A and B, based on their expected values.
    - Based on your approximation find the importance shift ratio β for A and B.
      β_A = 
      β_B = 
Make sure to first show the original reasoning, and then show the final approximate closed-form expressions.
please explicitly state its source, for example:  
> “From I. S. Gradshteyn and I. M. Ryzhik, Table of Integrals, Series, and Products, entry 3.310.3.”  
"""
#

def get_importance_form(symbolic_equation):
    if "min(" in symbolic_equation:
        prompt = f"""
{symbolic_equation}
C = min(A, B)
C is decided by random variable A and B.

SystemCapacity ≈ Throughput_Mbps.

Importance of feature A is defined as the expected absolute change in C when A is perturbed (similarly for B).

Suppose in the original environment:
E[A] = a, E[B] = b
Importance: X for A, Y for B.

New environment
E[A] = a', E[B] = b'
Importance: X' for A, Y' for B.

I need a importance shift factor β_A = X'/X and β_B = Y'/Y

Please follow these steps in your response:
1. First, explain the original mathematical formulation based on probability:
    - C = 
    - X = 
    - Y = 
    - In general, it would involve the CDF of (A−B) evaluated at 0, i.e., P(A < B) = F_D(0).
2. Under the assumption that the distribution of A and B are same, approximate this formulation to obtain a closed-form expression without using CDF:
    - Prefer using a ratio-based approximation between A and B, based on their expected values.
    - Based on your approximation find the importance shift ratio β for A and B.
      β_A = 
      β_B = 
Make sure to first show the original reasoning, and then show the final approximate closed-form expressions.
please explicitly state its source, for example:  
> “From I. S. Gradshteyn and I. M. Ryzhik, Table of Integrals, Series, and Products, entry 3.310.3.”  

Output format
1. explain the original mathematical formulation based on probability
2. Show the β_A, β_B
"""

    elif "/" in symbolic_equation:
        prompt = f"""
{symbolic_equation}
C = A/B
C is decided by random variable A and B.

SystemCapacity ≈ Throughput_Mbps.

Importance of feature A is defined as the expected absolute change in C when A is perturbed (similarly for B).

Suppose in the original environment:
E[A] = a, E[B] = b
Importance: X for A, Y for B.

New environment
E[A] = a', E[B] = b'
Importance: X' for A, Y' for B.

I need a importance shift factor β_A = X'/X and β_B = Y'/Y

Please follow these steps in your response:
1. First, explain the original mathematical of deriving X, X' and Y, Y'
2. Then, approximate this formulation to obtain a closed-form expression without using CDF:
      β_A = 
      β_B = 

Make sure to first show the original reasoning, and then show the final approximate closed-form expressions."""

    else:
        prompt = ""
        print(symbolic_equation)

    return prompt