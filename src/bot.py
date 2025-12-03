from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.messages.flat.QuickChatSelection import QuickChatSelection
from rlbot.utils.structures.game_data_struct import GameTickPacket

from util.ball_prediction_analysis import find_slice_at_time
from util.boost_pad_tracker import BoostPad, BoostPadTracker
from util.drive import steer_toward_target
from util.sequence import Sequence, ControlStep
from util.vec import Vec3


class MyBot(BaseAgent):

    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.active_sequence: Sequence = None
        self.boost_pad_tracker = BoostPadTracker()

    def initialize_agent(self):
        # Set up information about the boost pads now that the game is active and the info is available
        self.boost_pad_tracker.initialize_boosts(self.get_field_info())

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        """
        This function will be called by the framework many times per second. This is where you can
        see the motion of the ball, etc. and return controls to drive your car.
        """

        # Keep our boost pad info updated with which pads are currently active
        self.boost_pad_tracker.update_boost_status(packet)

        # This is good to keep at the beginning of get_output. It will allow you to continue
        # any sequences that you may have started during a previous call to get_output.
        if self.active_sequence is not None and not self.active_sequence.done:
            controls = self.active_sequence.tick(packet)
            if controls is not None:
                return controls

        # Gather some information about our car and the ball
        my_car = packet.game_cars[self.index]
        car_location = Vec3(my_car.physics.location)
        car_velocity = Vec3(my_car.physics.velocity)
        ball_location = Vec3(packet.game_ball.physics.location)

        ball_prediction = self.get_ball_prediction_struct()
        ball_in_future = find_slice_at_time(ball_prediction, packet.game_info.seconds_elapsed + 1.5)

        target_location = self.choose_target(packet, car_location, ball_location, ball_in_future)

        # Draw some things to help understand what the bot is thinking
        self.renderer.draw_line_3d(car_location, target_location, self.renderer.white())
        self.renderer.draw_string_3d(car_location, 1, 1, f'Speed: {car_velocity.length():.1f}', self.renderer.white())
        self.renderer.draw_rect_3d(target_location, 8, 8, True, self.renderer.cyan(), centered=True)

        if 750 < car_velocity.length() < 800:
            # We'll do a front flip if the car is moving at a certain speed.
            return self.begin_front_flip(packet)

        controls = SimpleControllerState()
        controls.steer = steer_toward_target(my_car, target_location)
        controls.throttle = 1.0
        controls.boost = self.should_boost(my_car, target_location)

        return controls

    def begin_front_flip(self, packet):
        # Send some quickchat just for fun
        self.send_quick_chat(team_only=False, quick_chat=QuickChatSelection.Information_IGotIt)

        # Do a front flip. We will be committed to this for a few seconds and the bot will ignore other
        # logic during that time because we are setting the active_sequence.
        self.active_sequence = Sequence([
            ControlStep(duration=0.05, controls=SimpleControllerState(jump=True)),
            ControlStep(duration=0.05, controls=SimpleControllerState(jump=False)),
            ControlStep(duration=0.2, controls=SimpleControllerState(jump=True, pitch=-1)),
            ControlStep(duration=0.8, controls=SimpleControllerState()),
        ])

        # Return the controls associated with the beginning of the sequence so we can start right away.
        return self.active_sequence.tick(packet)

    def choose_target(self, packet: GameTickPacket, car_location: Vec3, ball_location: Vec3, ball_in_future) -> Vec3:
        """Decide where to drive next based on kickoff, boost needs, and defense."""

        if self.is_kickoff(packet, ball_location):
            return ball_location

        my_car = packet.game_cars[self.index]
        my_goal_y = -5120 if my_car.team == 0 else 5120
        own_goal = Vec3(0, my_goal_y, 0)

        # Try to grab a nearby full boost if we're starving and not right next to the ball
        if my_car.boost < 25 and car_location.dist(ball_location) > 800:
            boost_location = self.find_nearest_boost(car_location)
            if boost_location is not None:
                return boost_location

        # Fall back to defending if the ball is dangerously close to our goal
        if abs(ball_location.y - my_goal_y) < 1500 and ball_location.dist(car_location) > 500:
            return own_goal

        if ball_in_future is not None:
            future_ball_location = Vec3(ball_in_future.physics.location)
            self.renderer.draw_line_3d(ball_location, future_ball_location, self.renderer.cyan())
            return future_ball_location

        return ball_location

    def is_kickoff(self, packet: GameTickPacket, ball_location: Vec3) -> bool:
        ball_stationary = packet.game_ball.physics.velocity.x == 0 and packet.game_ball.physics.velocity.y == 0
        return packet.game_info.is_round_active and ball_stationary and abs(ball_location.x) < 50 and abs(ball_location.y) < 50

    def find_nearest_boost(self, car_location: Vec3) -> Vec3:
        def pad_sort_key(pad: BoostPad):
            return car_location.dist(pad.location)

        active_pads = [pad for pad in self.boost_pad_tracker.get_full_boosts() if pad.is_active]
        if not active_pads:
            active_pads = [pad for pad in self.boost_pad_tracker.boost_pads if pad.is_active]

        if not active_pads:
            return None

        return min(active_pads, key=pad_sort_key).location

    def should_boost(self, my_car, target_location: Vec3) -> bool:
        velocity = Vec3(my_car.physics.velocity)
        needs_speed = velocity.length() < 2200 and my_car.boost > 0
        facing_target = steer_toward_target(my_car, target_location) == 0
        return needs_speed and facing_target
