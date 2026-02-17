"""
agentic_ai.py — LLM Orchestrator with Tool-Calling (Pillar 4)
===============================================================

An agentic AI system that uses an LLM (OpenAI GPT-4o) to orchestrate
all the other modules. The LLM acts as a natural-language interface,
calling tools (forecaster, optimizer, RL agent, heuristics) to answer
user questions about charging strategy.

Modes:
1. LIVE MODE:  Uses OpenAI API (requires OPENAI_API_KEY env variable)
2. MOCK MODE:  Simulates tool-calling without any API key (for offline
               demos and grading). Students see the full agentic pattern
               regardless of API access.

IEOR E4010: AI for Operations Research and Financial Engineering
Columbia University, Spring 2026
"""

import os
import json
import numpy as np
from typing import Dict, Any, Optional, List, Callable

from config import Config, DEFAULT_CONFIG

# Try importing OpenAI
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ============================================================
# Tool Definitions (OpenAI Function Calling Schema)
# ============================================================
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_price_forecast",
            "description": (
                "Get electricity price forecast for the next 24 hours "
                "using ML (XGBoost) and/or DL (LSTM) models. Returns predicted "
                "prices in $/MWh for each hour."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model_type": {
                        "type": "string",
                        "enum": ["ml", "dl", "both"],
                        "description": "Which forecasting model to use",
                    }
                },
                "required": ["model_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "solve_lp_optimal",
            "description": (
                "Solve for the optimal charging schedule using Linear Programming. "
                "Assumes perfect knowledge of future prices. Returns the minimum-cost "
                "schedule with V2G arbitrage."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "allow_v2g": {
                        "type": "boolean",
                        "description": "Whether to allow vehicle-to-grid discharging",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_rl_policy",
            "description": (
                "Run the trained RL (PPO) agent to generate a charging schedule. "
                "Unlike LP, the RL agent does not know future prices — it decides "
                "based only on current observations."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_heuristic",
            "description": (
                "Run a simple rule-based charging strategy. "
                "Options: 'asap' (charge immediately), 'alap' (delay as long as possible), "
                "'round_robin' (rotate charging slots)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "string",
                        "enum": ["asap", "alap", "round_robin"],
                        "description": "Which heuristic strategy to run",
                    }
                },
                "required": ["strategy"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_strategies",
            "description": (
                "Compare all strategies (heuristics, LP optimal, RL) side by side. "
                "Returns a table of costs, constraint violations, and compute times."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_schedule",
            "description": (
                "Explain a charging schedule in plain language. Takes the output "
                "of any strategy and generates a human-readable summary."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "strategy_name": {
                        "type": "string",
                        "description": "Name of the strategy to explain",
                    }
                },
                "required": ["strategy_name"],
            },
        },
    },
]


# ============================================================
# Tool Executor
# ============================================================
class ToolExecutor:
    """Executes tools called by the LLM agent.

    Each tool calls the actual module functions and returns results
    as structured data that the LLM can interpret.
    """

    def __init__(
        self,
        cfg: Config = DEFAULT_CONFIG,
        schedules=None,
        price_curve=None,
        ml_model=None,
        lstm_dict=None,
        rl_model=None,
        prices_df=None,
    ):
        self.cfg = cfg
        self.schedules = schedules
        self.price_curve = price_curve
        self.ml_model = ml_model
        self.lstm_dict = lstm_dict
        self.rl_model = rl_model
        self.prices_df = prices_df
        self._cached_results: Dict[str, Any] = {}

    def execute(self, tool_name: str, arguments: Dict) -> str:
        """Execute a tool and return results as JSON string."""
        try:
            if tool_name == "get_price_forecast":
                return self._get_price_forecast(arguments.get("model_type", "both"))
            elif tool_name == "solve_lp_optimal":
                return self._solve_lp(arguments.get("allow_v2g", True))
            elif tool_name == "run_rl_policy":
                return self._run_rl()
            elif tool_name == "run_heuristic":
                return self._run_heuristic(arguments.get("strategy", "round_robin"))
            elif tool_name == "compare_strategies":
                return self._compare_all()
            elif tool_name == "explain_schedule":
                return self._explain(arguments.get("strategy_name", "lp_optimal"))
            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _get_price_forecast(self, model_type: str) -> str:
        from ml_forecaster import predict_next_24h
        from dl_forecaster import predict_next_24h_lstm

        results = {}

        if self.prices_df is not None:
            recent = self.prices_df["price_mwh"].values[-200:]
        else:
            recent = np.random.uniform(20, 60, 200)

        if model_type in ("ml", "both") and self.ml_model is not None:
            try:
                ml_pred = predict_next_24h(self.ml_model, recent, 18, 2, 6, self.cfg)
                results["ml_forecast"] = {
                    "prices": [round(p, 1) for p in ml_pred],
                    "min_price": round(float(ml_pred.min()), 1),
                    "max_price": round(float(ml_pred.max()), 1),
                    "min_price_hour": int(np.argmin(ml_pred)),
                    "max_price_hour": int(np.argmax(ml_pred)),
                }
            except Exception as e:
                results["ml_forecast"] = {"error": str(e)}

        if model_type in ("dl", "both") and self.lstm_dict is not None:
            try:
                dl_pred = predict_next_24h_lstm(self.lstm_dict, recent)
                results["dl_forecast"] = {
                    "prices": [round(p, 1) for p in dl_pred],
                    "min_price": round(float(dl_pred.min()), 1),
                    "max_price": round(float(dl_pred.max()), 1),
                    "min_price_hour": int(np.argmin(dl_pred)),
                    "max_price_hour": int(np.argmax(dl_pred)),
                }
            except Exception as e:
                results["dl_forecast"] = {"error": str(e)}

        if not results:
            # Fallback: use actual price curve if available
            if self.price_curve is not None:
                prices = self.price_curve[:24]
                results["actual_prices"] = {
                    "prices": [round(p, 1) for p in prices],
                    "min_price": round(float(prices.min()), 1),
                    "max_price": round(float(prices.max()), 1),
                    "note": "Using actual prices (no trained model provided)",
                }
            else:
                results["error"] = "No forecasting model available"

        return json.dumps(results)

    def _solve_lp(self, allow_v2g: bool = True) -> str:
        from optimizer import solve_optimal_schedule

        if self.schedules is None or self.price_curve is None:
            return json.dumps({"error": "Schedules and price curve required"})

        result = solve_optimal_schedule(
            self.schedules, self.price_curve, self.cfg,
            allow_v2g=allow_v2g, verbose=False
        )
        self._cached_results["lp_optimal"] = result

        return json.dumps({
            "strategy": "lp_optimal",
            "net_cost": round(result["net_cost"], 2),
            "charging_cost": round(result["total_cost"], 2),
            "v2g_revenue": round(result["v2g_revenue"], 2),
            "degradation_cost": round(result["degradation_cost"], 2),
            "max_power_kw": round(result["max_power_kw"], 1),
            "evs_meeting_target": result["evs_meeting_target"],
            "total_evs": result["total_evs"],
            "status": result["status"],
        })

    def _run_rl(self) -> str:
        if self.rl_model is None:
            return json.dumps({"error": "No trained RL model provided"})

        from rl_agent import run_single_episode
        result = run_single_episode(
            self.rl_model, self.cfg,
            schedules=self.schedules, price_curve=self.price_curve
        )
        self._cached_results["ppo_rl"] = result

        return json.dumps({
            "strategy": "ppo_rl",
            "net_cost": round(result["net_cost"], 2),
            "total_cost": round(result["total_cost"], 2),
            "v2g_revenue": round(result["v2g_revenue"], 2),
            "degradation_cost": round(result["degradation_cost"], 2),
            "penalties": round(result["penalties"], 2),
            "evs_meeting_target": result["evs_meeting_target"],
            "total_evs": result["total_evs"],
        })

    def _run_heuristic(self, strategy: str) -> str:
        from heuristics import run_heuristic as _run_h
        from environment import make_env

        env = make_env(self.cfg, schedules=self.schedules, price_curve=self.price_curve)
        result = _run_h(env, strategy=strategy)
        self._cached_results[strategy] = result

        return json.dumps({
            "strategy": strategy,
            "net_cost": round(result["net_cost"], 2),
            "total_cost": round(result["total_cost"], 2),
            "v2g_revenue": round(result["v2g_revenue"], 2),
            "penalties": round(result["penalties"], 2),
            "evs_meeting_target": result["evs_meeting_target"],
            "total_evs": result["total_evs"],
        })

    def _compare_all(self) -> str:
        rows = []
        for name in ["asap", "alap", "round_robin", "lp_optimal", "ppo_rl"]:
            if name not in self._cached_results:
                # Run it
                if name in ("asap", "alap", "round_robin"):
                    self._run_heuristic(name)
                elif name == "lp_optimal":
                    self._solve_lp()
                elif name == "ppo_rl" and self.rl_model is not None:
                    self._run_rl()

            if name in self._cached_results:
                r = self._cached_results[name]
                rows.append({
                    "strategy": name,
                    "net_cost": round(r.get("net_cost", r.get("net_cost", 0)), 2),
                    "evs_at_target": r.get("evs_meeting_target", "N/A"),
                    "v2g_revenue": round(r.get("v2g_revenue", 0), 2),
                    "penalties": round(r.get("penalties", 0), 2),
                })

        return json.dumps({"comparison": rows})

    def _explain(self, strategy_name: str) -> str:
        if strategy_name not in self._cached_results:
            return json.dumps({"error": f"Run {strategy_name} first"})

        r = self._cached_results[strategy_name]
        cfg = self.cfg

        explanation = {
            "strategy": strategy_name,
            "summary": (
                f"The {strategy_name} strategy manages {cfg.fleet.num_evs} EVs "
                f"over {cfg.time.simulation_hours} hours. "
                f"Net electricity cost: ${r.get('net_cost', 0):.2f}. "
                f"{r.get('evs_meeting_target', 0)}/{r.get('total_evs', cfg.fleet.num_evs)} "
                f"EVs met their {cfg.fleet.target_soc:.0%} departure target. "
                f"V2G revenue: ${r.get('v2g_revenue', 0):.2f}."
            ),
            "recommendation": (
                "Charge primarily during off-peak hours (midnight-5AM) when prices "
                "are lowest. If V2G is enabled, discharge EVs with high SoC during "
                "the evening peak (5-9PM) to earn revenue, but ensure all EVs still "
                "reach 90% by their departure time."
            ),
        }

        return json.dumps(explanation)


# ============================================================
# LLM Agent (OpenAI)
# ============================================================
SYSTEM_PROMPT = """You are an EV Smart Charging Assistant for a fleet depot operator.
You have access to tools that can forecast electricity prices, optimize charging schedules,
run RL policies, and compare strategies.

When a user asks about charging, you should:
1. Call the relevant tools to get data
2. Interpret the results
3. Provide clear, actionable recommendations

Always explain your reasoning in terms the operator understands:
- Cost savings in dollars
- Which EVs to charge when
- Whether V2G (selling back to grid) is profitable
- Whether all departure deadlines will be met

Be concise but thorough. Use specific numbers from the tool results."""


class EVChargingAgent:
    """Agentic AI that orchestrates all modules via LLM tool-calling.

    Supports:
    - Live mode (OpenAI API)
    - Mock mode (no API key needed)
    """

    def __init__(
        self,
        cfg: Config = DEFAULT_CONFIG,
        tool_executor: Optional[ToolExecutor] = None,
        use_mock: Optional[bool] = None,
    ):
        """Initialize the agent.

        Args:
            cfg:            Configuration
            tool_executor:  ToolExecutor with loaded models
            use_mock:       Force mock mode (None = auto-detect)
        """
        self.cfg = cfg
        self.tool_executor = tool_executor or ToolExecutor(cfg)

        # Auto-detect mode
        if use_mock is None:
            api_key = os.environ.get(cfg.agent.api_key_env_var, "")
            self.use_mock = not (HAS_OPENAI and api_key)
        else:
            self.use_mock = use_mock

        if not self.use_mock:
            self.client = OpenAI()
            self.model = cfg.agent.model
        else:
            self.client = None

        self.conversation_history: List[Dict] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        mode_str = "MOCK" if self.use_mock else f"LIVE ({cfg.agent.model})"
        print(f"EV Charging Agent initialized in {mode_str} mode")

    def chat(self, user_message: str) -> str:
        """Send a message and get a response.

        The LLM may call tools automatically to gather data
        before generating its final response.

        Args:
            user_message: Natural language query

        Returns:
            Agent's response string
        """
        self.conversation_history.append({"role": "user", "content": user_message})

        if self.use_mock:
            return self._mock_response(user_message)
        else:
            return self._live_response()

    def _live_response(self) -> str:
        """Get response from OpenAI API with tool-calling."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation_history,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
            temperature=self.cfg.agent.temperature,
            max_tokens=self.cfg.agent.max_tokens,
        )

        message = response.choices[0].message

        # Handle tool calls
        max_rounds = 5
        round_count = 0
        while message.tool_calls and round_count < max_rounds:
            round_count += 1
            self.conversation_history.append(message.model_dump())

            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                print(f"  → Calling tool: {func_name}({args})")
                result = self.tool_executor.execute(func_name, args)

                self.conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

            # Get next response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                temperature=self.cfg.agent.temperature,
                max_tokens=self.cfg.agent.max_tokens,
            )
            message = response.choices[0].message

        # Final text response
        assistant_msg = message.content or ""
        self.conversation_history.append({"role": "assistant", "content": assistant_msg})
        return assistant_msg

    def _mock_response(self, user_message: str) -> str:
        """Generate a mock response by auto-calling relevant tools.

        Simulates the agentic pattern without an actual LLM.
        Students can see how tool-calling works even without API access.
        """
        msg_lower = user_message.lower()
        tool_results = []

        # Determine which tools to call based on keywords
        if any(w in msg_lower for w in ["forecast", "price", "predict", "tomorrow"]):
            print("  → [Mock] Calling tool: get_price_forecast({'model_type': 'both'})")
            result = self.tool_executor.execute("get_price_forecast", {"model_type": "both"})
            tool_results.append(("get_price_forecast", result))

        if any(w in msg_lower for w in ["optimal", "best", "cheapest", "minimize", "lp"]):
            print("  → [Mock] Calling tool: solve_lp_optimal({'allow_v2g': true})")
            result = self.tool_executor.execute("solve_lp_optimal", {"allow_v2g": True})
            tool_results.append(("solve_lp_optimal", result))

        if any(w in msg_lower for w in ["rl", "agent", "learn", "policy"]):
            print("  → [Mock] Calling tool: run_rl_policy()")
            result = self.tool_executor.execute("run_rl_policy", {})
            tool_results.append(("run_rl_policy", result))

        if any(w in msg_lower for w in ["compare", "all", "versus", "vs"]):
            print("  → [Mock] Calling tool: compare_strategies()")
            result = self.tool_executor.execute("compare_strategies", {})
            tool_results.append(("compare_strategies", result))

        if any(w in msg_lower for w in ["heuristic", "simple", "baseline", "asap", "round"]):
            strategy = "round_robin"
            if "asap" in msg_lower:
                strategy = "asap"
            elif "alap" in msg_lower:
                strategy = "alap"
            print(f"  → [Mock] Calling tool: run_heuristic({{strategy: '{strategy}'}})")
            result = self.tool_executor.execute("run_heuristic", {"strategy": strategy})
            tool_results.append(("run_heuristic", result))

        # If no tools matched, run a comparison by default
        if not tool_results:
            print("  → [Mock] Calling tool: solve_lp_optimal({'allow_v2g': true})")
            result = self.tool_executor.execute("solve_lp_optimal", {"allow_v2g": True})
            tool_results.append(("solve_lp_optimal", result))

        # Build mock response from tool results
        response_parts = [f"[Mock Mode — showing tool-calling pattern]\n"]
        response_parts.append(f"User question: \"{user_message}\"\n")
        response_parts.append("Tools called:")

        for tool_name, result_json in tool_results:
            try:
                data = json.loads(result_json)
                response_parts.append(f"\n  {tool_name}():")

                if tool_name == "solve_lp_optimal":
                    response_parts.append(
                        f"    Net cost: ${data.get('net_cost', 'N/A')}, "
                        f"V2G revenue: ${data.get('v2g_revenue', 'N/A')}, "
                        f"EVs at target: {data.get('evs_meeting_target')}/{data.get('total_evs')}"
                    )
                elif tool_name == "get_price_forecast":
                    for model_key in ["ml_forecast", "dl_forecast", "actual_prices"]:
                        if model_key in data:
                            f = data[model_key]
                            response_parts.append(
                                f"    {model_key}: range ${f.get('min_price', '?')}"
                                f"–${f.get('max_price', '?')}/MWh"
                            )
                elif tool_name == "compare_strategies":
                    for row in data.get("comparison", []):
                        response_parts.append(
                            f"    {row['strategy']:15s}: ${row['net_cost']:8.2f} | "
                            f"Targets: {row['evs_at_target']}"
                        )
                else:
                    response_parts.append(f"    {json.dumps(data, indent=2)[:200]}")
            except json.JSONDecodeError:
                response_parts.append(f"    [Error parsing result]")

        response_parts.append(
            "\n[In live mode, GPT-4o would synthesize these results into "
            "a natural language recommendation for the fleet operator.]"
        )

        response = "\n".join(response_parts)
        self.conversation_history.append({"role": "assistant", "content": response})
        return response


# ============================================================
# Convenience function
# ============================================================
def create_agent(
    cfg: Config = DEFAULT_CONFIG,
    schedules=None,
    price_curve=None,
    ml_model=None,
    lstm_dict=None,
    rl_model=None,
    prices_df=None,
    use_mock: Optional[bool] = None,
) -> EVChargingAgent:
    """Create a fully-configured agent.

    Args:
        cfg:          Configuration
        schedules:    EV schedules for the scenario
        price_curve:  Price curve for the scenario
        ml_model:     Trained ML model
        lstm_dict:    Trained LSTM dict
        rl_model:     Trained RL model (or path)
        prices_df:    Historical prices DataFrame
        use_mock:     Force mock mode (None = auto-detect from API key)

    Returns:
        EVChargingAgent instance
    """
    executor = ToolExecutor(
        cfg=cfg,
        schedules=schedules,
        price_curve=price_curve,
        ml_model=ml_model,
        lstm_dict=lstm_dict,
        rl_model=rl_model,
        prices_df=prices_df,
    )
    return EVChargingAgent(cfg=cfg, tool_executor=executor, use_mock=use_mock)


# ============================================================
# Interactive Chat Loop
# ============================================================
def interactive_chat(agent: EVChargingAgent):
    """Run an interactive chat loop with the agent."""
    print("\n" + "=" * 60)
    print("EV Smart Charging Assistant")
    print("=" * 60)
    print("Ask me about charging strategies, price forecasts, or")
    print("how to minimize your electricity costs.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not user_input:
            continue

        response = agent.chat(user_input)
        print(f"\nAssistant: {response}\n")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    from data_utils import generate_synthetic_prices, get_daily_price_curve, generate_ev_schedules

    cfg = Config()
    prices_df = generate_synthetic_prices(cfg, num_days=30)
    price_curve = get_daily_price_curve(prices_df, day_index=0, cfg=cfg)
    schedules = generate_ev_schedules(cfg)

    agent = create_agent(
        cfg=cfg,
        schedules=schedules,
        price_curve=price_curve,
        prices_df=prices_df,
        use_mock=None,  # Auto-detect
    )

    # Demo queries
    demo_queries = [
        "What's the cheapest charging plan for tonight?",
        "Compare all strategies for me.",
        "Should I use V2G to sell energy back during peak hours?",
    ]

    for query in demo_queries:
        print(f"\n{'='*60}")
        print(f"User: {query}")
        print("-" * 60)
        response = agent.chat(query)
        print(f"\nAssistant: {response}")
