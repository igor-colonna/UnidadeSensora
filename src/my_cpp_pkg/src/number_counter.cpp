#include "rclcpp/rclcpp.hpp"
#include "example_interfaces/msg/int64.hpp"
#include "example_interfaces/srv/set_bool.hpp"

using namespace std::chrono_literals;
using namespace std::placeholders;


class NumberCounterNode : public rclcpp::Node{

    public:
        NumberCounterNode() : Node("number_counter"){
            publisher_ = this->create_publisher<example_interfaces::msg::Int64>("number_count", 10);
            subscriber_ = this->create_subscription<example_interfaces::msg::Int64>("number", 10, 
                std::bind(&NumberCounterNode::callbackNumberReceived, this, _1));
            reset_counter_srv_ = this->create_service<example_interfaces::srv::SetBool>("reset_counter", 
                std::bind(&NumberCounterNode::callbackResetCounter, this, _1, _2));
            RCLCPP_INFO(this->get_logger(), "Number Counter has been started");
        }

    private:
        void callbackNumberReceived(const example_interfaces::msg::Int64::SharedPtr msg){
            counter_ += msg->data;
            auto newMsg = example_interfaces::msg::Int64();
            newMsg.data = counter_;
            publisher_->publish(newMsg);
            //RCLCPP_INFO(this->get_logger(), "%i", msg->data)
        }

        void callbackResetCounter(const example_interfaces::srv::SetBool::Request::SharedPtr request,
                                  const example_interfaces::srv::SetBool::Response::SharedPtr response)
        {
            if(request->data)
            {
                counter_ = 0;
                response->success = true;
                response->message = "Counter has been reset";
            }   
            else
            {
                response->success = false;
                response->message = "Counter has not been reset";
            } 

        }

        int counter_;
        rclcpp::Publisher<example_interfaces::msg::Int64>::SharedPtr publisher_;
        rclcpp::Subscription<example_interfaces::msg::Int64>::SharedPtr subscriber_;
        rclcpp::Service<example_interfaces::srv::SetBool>::SharedPtr reset_counter_srv_;
};

int main(int argc, char **argv){
    
    rclcpp::init(argc, argv);
    auto node = std::make_shared<NumberCounterNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}