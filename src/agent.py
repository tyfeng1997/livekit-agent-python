import logging

from dotenv import load_dotenv
from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentFalseInterruptionEvent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
    ChatContext,
    ChatMessage
)
from livekit.agents.llm import function_tool
from livekit.plugins import cartesia, deepgram, noise_cancellation, openai, silero
from livekit.plugins import anthropic 
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.agents import UserStateChangedEvent, AgentStateChangedEvent
from livekit.plugins import silero
from livekit.plugins import assemblyai
from livekit.plugins import google

logger = logging.getLogger("agent")

load_dotenv(".env.local")


 
class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            # instructions="""You are a helpful voice AI assistant.
            # You eagerly assist users with their questions by providing information from your extensive knowledge.
            # Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            # You are curious, friendly, and have a sense of humor.""",
            instructions="You are a helpful AI assistant. Keep your responses concise and conversational. You're having a real-time voice conversation, so avoid long explanations unless asked."
        )
    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Greet the user with a warm welcome"
        )
    async def on_exit(self):
        await self.session.generate_reply(
            instructions="Tell the user a friendly goodbye before you exit.",
        )
    
    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage,
    ) -> None:
        # rag_content = await my_rag_lookup(new_message.text_content())
        # turn_ctx.add_message(
        #     role="assistant", 
        #     content=f"Additional information relevant to the user's next message: {rag_content}"
        # )
        pass
    
    # all functions annotated with @function_tool will be passed to the LLM when this
    # agent is active
    @function_tool
    async def lookup_weather(self, context: RunContext, location: str):
        """Use this tool to look up current weather information in the given location.

        If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.

        Args:
            location: The location to look up weather information for (e.g. city name)
        """

        logger.info(f"Looking up weather for {location}")

        return "sunny with a temperature of 70 degrees."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, Deepgram, and the LiveKit turn detector
    session = AgentSession(
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all providers at https://docs.livekit.io/agents/integrations/llm/
        # llm=openai.LLM(model="gpt-4o-mini"),
        llm=anthropic.LLM(model="claude-sonnet-4-20250514"),
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all providers at https://docs.livekit.io/agents/integrations/stt/
        # stt=deepgram.STT(model="nova-3", language="multi"),
        # stt=cartesia.STT(model="ink-whisper",language="zh"),
        turn_detection="stt",
        stt=assemblyai.STT(
            end_of_turn_confidence_threshold=0.7,
            min_end_of_turn_silence_when_confident=160,
            max_turn_silence=2400,
        ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all providers at https://docs.livekit.io/agents/integrations/tts/
        tts=cartesia.TTS(voice="a0e99841-438c-4a64-b679-ae501e7d6091"),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        # turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
        
    )

    

    # sometimes background noise could interrupt the agent session, these are considered false positive interruptions
    # when it's detected, you may resume the agent's speech
    @session.on("agent_false_interruption")
    def _on_agent_false_interruption(ev: AgentFalseInterruptionEvent):
        logger.info("false positive interruption, resuming")
        session.generate_reply(instructions=ev.extra_instructions or NOT_GIVEN)

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)
        
    @session.on("user_state_changed")
    def on_user_state_changed(ev: UserStateChangedEvent):
        if ev.new_state == "speaking":
            print("User started speaking")
        elif ev.new_state == "listening":
            print("User stopped speaking")
        elif ev.new_state == "away":
            print("User is not present (e.g. disconnected)")

    @session.on("agent_state_changed")
    def on_agent_state_changed(ev: AgentStateChangedEvent):
        if ev.new_state == "initializing":
            print("Agent is starting up")
        elif ev.new_state == "idle":
            print("Agent is ready but not processing")
        elif ev.new_state == "listening":
            print("Agent is listening for user input")
        elif ev.new_state == "thinking":
            print("Agent is processing user input and generating a response")
        elif ev.new_state == "speaking":
            print("Agent started speaking")
            
        

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/integrations/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/integrations/avatar/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    import os
    agent_name = os.getenv("AGENT_NAME", "")
    print(f"Starting agent with name: {agent_name}")
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint, 
            prewarm_fnc=prewarm,
            agent_name=agent_name
        )
    )
