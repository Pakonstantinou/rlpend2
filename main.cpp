#include <cstdlib>
#include <iostream>
#include <cmath>
#include <robot_dart/control/simple_control.hpp>
#include <robot_dart/robot_dart_simu.hpp>
#include <torch/torch.h>
#include <memory>
#include <iomanip>
#include <math.h>       /* sin */
#include <random>
#include "Net.h"
#ifdef GRAPHIC
#include <robot_dart/gui/magnum/graphics.hpp>
#endif
auto cuda_available = torch::cuda::is_available();
torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
std::default_random_engine generator;

namespace rd = robot_dart;

class pendulum {

public:
    torch::Tensor _state;
    //bool done;
    //double reward;
    int simusteps=0;
    std::shared_ptr<robot_dart::Robot> robot = std::make_shared<robot_dart::Robot>("pendulum.urdf");
    std::shared_ptr<robot_dart::RobotDARTSimu> simu = std::make_shared<robot_dart::RobotDARTSimu>(0.001);

    pendulum()
    {
        robot->fix_to_world();
        robot->set_position_enforced(true);
        robot->set_positions(robot_dart::make_vector({M_PI}));
        robot->set_actuator_types("torque");
        simu->add_robot(robot);
#ifdef GRAPHIC
        auto graphics = std::make_shared<robot_dart::gui::magnum::Graphics>();
            simu->set_graphics(graphics);
#endif
        //_state = torch::tensor({M_PI}, torch::kDouble);
    }

    std::tuple<torch::Tensor, double, bool> step(torch::Tensor act)
    {
        bool success;
        auto act_a = act.accessor<double, 1>();
        double move = act_a[0];

        auto cmds = rd::make_vector({move});

        for (int i = 0; i < 50; i++) {
            robot->set_commands(cmds);
            simu->step_world();
            simusteps++;
        }

        double temp_pos = robot->positions()[0];
        //double data[] = {temp_pos};
        double sin_pos=sin(robot->positions()[0]);
        double cos_pos=cos(robot->positions()[0]);
        bool done = false;
        double reward;
        double temp_velocity = robot->velocities()[0];
        double theta = angle_dist(temp_pos, 0.);
        // reward=-(std::abs(M_PI-robot->positions()[0]));
        reward = -theta;
        //reward = -std::abs(temp_velocity);

        //if (std::abs(M_PI-temp_pos)<0.0001) {
        if ((theta)<0.1){
            //if ((std::abs(theta)<0.1)){

            //auto cmds = rd::make_vector({0});
            //robot->set_commands(cmds);
            done = false;
            reward = 10;
            //simusteps = 0;
            //std::cout << "success"<<std::endl;
            success=true;
            //std::cout<<temp_pos<<std::endl;

            //torch::Tensor reset();
        }
        if (simusteps == 5000) {
            //auto cmds = rd::make_vector({0});
            //robot->set_commands(cmds);
            done = true;
            simusteps = 0;
            std::cout << "fail"<<std::endl;
            success=false;
            std::cout<<theta<<std::endl;
            //torch::Tensor reset();
        }

        //_state = torch::from_blob(data, {1}, torch::TensorOptions().dtype(torch::kDouble));
        _state = torch::tensor({sin_pos, cos_pos, temp_velocity}, torch::kDouble);

        auto _stateNan = at::isnan(_state).any().item<bool>();

        if (_stateNan==1){
            std::cout<<"_statesNan"<<_state<<std::endl;
            std::cout<<"mu nan"<<std::endl;

            exit (EXIT_FAILURE);
        }


        return {_state, reward, done};
    }

    torch::Tensor reset()
    {
        simusteps = 0;
        robot->reset();
        robot->set_positions(robot_dart::make_vector({M_PI}));
        //double tempor =robot->positions()[0];
        _state = torch::tensor({0.0,-1.0,0.0}, torch::kDouble);
        return _state;
    }

    static double angle_dist(double a, double b)
    {
        double theta = b - a;
        while (theta < -M_PI)
            theta += 2 * M_PI;
        while (theta > M_PI)
            theta -= 2 * M_PI;
        return std::abs(theta);
    }
};

class Agent{
public:
    float gamma = 0.99;
    int n_outputs = 1;
    int n_actions = 2;
    NeuralNet actor = nullptr;
    NeuralNet critic = nullptr;
    int layer1_size = 64;
    int layer2_size = 64;
    torch::Tensor log_probs;
    torch::optim::Adam *actor_optimizer;
    torch::optim::Adam *critic_optimizer;

    Agent(float alpha, float beta, int input_dims){
        this->actor = NeuralNet(input_dims, layer1_size, n_actions);
        this->actor->to(device);
        this->actor->to(torch::kDouble);

        critic = NeuralNet(input_dims, layer1_size, 1);
        critic->to(device);
        critic->to(torch::kDouble);

        // Optimizer
        actor_optimizer = new torch::optim::Adam(actor->parameters(), torch::optim::AdamOptions(alpha));
        critic_optimizer = new torch::optim::Adam(critic->parameters(), torch::optim::AdamOptions(beta));
    }

    torch::Tensor choose_action(torch::Tensor observation){
        torch::Tensor logsigma;
        //std::cout<<observation<<std::endl;
        // torch::Tensor test = torch::full({1, 3}, /*value=*/observation);
        //std::cout<<test<<std::endl;
        torch::Tensor tensor = torch::ones(5);
        torch::Tensor output = actor->forward(observation);
//        std::cout<<output;
//        std::cout<<output;


        torch::Tensor mu = output[0].to(torch::kDouble);
        logsigma = output[1].to(torch::kDouble); //add exp of sigma
        //std::cout<<mu<<logsigma<<std::endl;
        torch::Tensor sigma = torch::exp(logsigma).to(torch::kDouble);;
        //std::cout<<logsigma<<sigma<<std::endl;

        //std::normal_distribution<float> distribution(mu.item<float>(), sigma.item<float>());

        // auto sampler1 = torch::randn({1}) * sigma + mu ;
        // auto pdf = (1.0 / (sigma * std::sqrt(2.0 * M_PI))) * torch::exp(-0.5 * torch::pow((sampler1 - mu) / sigma, 2));
        // this->log_probs = torch::log(pdf);
        auto sample = torch::randn({1},torch::kDouble)*sigma + mu;

        // float action = tanh(sampler1.item<float>());
        auto pdf = (1.0 / (sigma * std::sqrt(2*M_PI))) * torch::exp(-0.5 * torch::pow((sample.detach() - mu) / sigma, 2));
        //auto probs = distribution(generator);
        //std::cout<<"pdf"<<pdf<<std::endl;
        this->log_probs = torch::log(pdf);
        //this->log_probs = torch::log(torch::tensor(probs));
        //std::cout<<probs<<log_probs<<std::endl;

        torch::Tensor action = torch::tanh(sample);

        return action * 5;
    }

    void learn(torch::Tensor state, double reward, torch::Tensor new_state, bool done){
        this->actor_optimizer->zero_grad();
        this->critic_optimizer->zero_grad();

        torch::Tensor critic_value_ = this->critic->forward(new_state);
        torch::Tensor critic_value = this->critic->forward(state);

        torch::Tensor tensor_reward = torch::tensor(reward);
        torch::Tensor delta = tensor_reward + this->gamma*critic_value_ * (1 * int(!done)) - critic_value;

        auto actor_loss = -1 * this->log_probs * delta;
        auto critic_loss = torch::pow(delta, 2);
        //std::cout<<log_probs<<std::endl<<delta<<std::endl;

        (actor_loss + critic_loss).backward();
        this->actor_optimizer->step();
        this->critic_optimizer->step();
    }
};

int main() {
    std::cout << "FeedForward Neural Network\n\n";

    // Hyper parameters
    const int64_t input_size = 1;
    const int64_t hidden_size = 256;
    const int64_t num_classes = 2;
    const int64_t batch_size = 100;
    const size_t num_epochs = 5;
    const double alpha = 0.000005;
    const double beta = 0.00001;

    pendulum env;

    Agent *agent = new Agent(0.000005, 0.00001, 3);

    int num_episodes = 800000;
    bool done;
    double score;
    double reward;

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Training...\n";

    // Train the model
    for (int i = 0; i < num_episodes; i++){
        done = false;
        score = 0;
        torch::Tensor observation = env.reset();

        while (!done){
            torch::Tensor action = agent->choose_action(observation);
            torch::Tensor observation_;
            std::tie(observation_, reward, done) = env.step(action);
            agent->learn(observation, reward, observation_, done);
            observation = observation_;
            score += reward;
        }

        printf("Episode %d score %.2f\n", i, score);
    }

    return 0;
}