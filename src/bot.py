import math

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.messages.flat.QuickChatSelection import QuickChatSelection
from rlbot.utils.structures.game_data_struct import GameTickPacket

from util.boost_pad_tracker import BoostPadTracker
from util.drive import steer_toward_target
from util.orientation import Orientation, relative_location
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
        ball_velocity = Vec3(packet.game_ball.physics.velocity)
        car_orientation = Orientation(my_car.physics.rotation)

        # Aim to carry the ball toward the opponent's goal by staying just behind it.
        opponent_goal_y = 5120 if self.team == 0 else -5120
        opponent_goal = Vec3(0, opponent_goal_y, 0)
        direction_to_goal = (opponent_goal - ball_location).flat()

        # Ensure we always have a direction to offset from, even if the ball is centered.
        if direction_to_goal.length() < 1:
            direction_to_goal = Vec3(0, 1 if opponent_goal_y > 0 else -1, 0)

        approach_offset = direction_to_goal.rescale(250)
        target_location = ball_location - approach_offset

        # Draw some things to help understand what the bot is thinking
        self.renderer.draw_line_3d(car_location, target_location, self.renderer.white())
        self.renderer.draw_string_3d(car_location, 1, 1, f'Speed: {car_velocity.length():.1f}', self.renderer.white())
        self.renderer.draw_rect_3d(target_location, 8, 8, True, self.renderer.cyan(), centered=True)
        self.renderer.draw_line_3d(ball_location, ball_location + Vec3(0, 0, 200), self.renderer.red())

        relative_target = relative_location(car_location, car_orientation, target_location)
        angle_to_target = math.atan2(relative_target.y, relative_target.x)
        distance_to_ball = car_location.flat().dist(ball_location.flat())

        controls = SimpleControllerState()
        controls.steer = steer_toward_target(my_car, target_location)

        # Stay under the ball: slow down as we get close so it can settle on the roof.
        if distance_to_ball > 2500:
            controls.throttle = 1.0
            controls.boost = abs(angle_to_target) < 0.3
        elif distance_to_ball > 1200:
            controls.throttle = 0.8
            controls.boost = abs(angle_to_target) < 0.35 and distance_to_ball > 1500
        else:
            controls.throttle = 0.35

        # Apply gentle steering help to stay aligned when making tight turns.
        controls.handbrake = abs(angle_to_target) > 1.8 and car_velocity.length() < 900

        # Nudge the ball up for a dribble when we're under it and it's low.
        relative_ball = relative_location(car_location, car_orientation, ball_location)
        low_and_centered = relative_ball.x > -30 and abs(relative_ball.y) < 150 and ball_location.z < 200
        if my_car.has_wheel_contact and distance_to_ball < 180 and ball_location.z < 120:
            controls.jump = True
        elif low_and_centered:
            controls.pitch = -0.15

        # Match ball speed when carrying it to reduce losing control.
        forward_speed = car_velocity.dot(car_orientation.forward)
        desired_speed = max(600, min(ball_velocity.length() + 400, 1400))
        speed_error = desired_speed - forward_speed
        if distance_to_ball < 1000:
            controls.throttle = max(min(speed_error / 1000, 0.75), -0.5)
            controls.boost = False

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
