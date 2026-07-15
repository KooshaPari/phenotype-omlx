// ZeroMQ fleet transport — C++ source compiled by the autotools/pkger
// when this crate is part of the perf-core workspace. Used as a transport
// for the fleet-proto Fleet message dispatch protocol.
//
// Build prerequisites:
//   * libzmq >= 4.3  (`brew install zeromq` / `apt install libzmq3-dev`)
//
// Wire protocol: a single zmq_msg_t per dispatch, length-prefixed.

#include <zmq.h>
#include "zeromq_transport.hpp"

namespace phenotype {

ZeroMqTransport::ZeroMqTransport(void* context, std::string_view endpoint, Role role)
    : context_(context), endpoint_(endpoint), role_(role) {
    socket_ = zmq_socket(static_cast<zmq_context_t*>(context_),
                         role == Role::PUSH ? ZMQ_PUSH : ZMQ_PULL);
    if (role == Role::PUSH) zmq_connect(socket_, endpoint.c_str());
    else                     zmq_bind  (socket_, endpoint.c_str());
}

ZeroMqTransport::~ZeroMqTransport() {
    if (socket_) zmq_close(socket_);
}

bool ZeroMqTransport::send(const FleetMessage& msg) {
    zmq_msg_t zmsg;
    zmq_msg_init_size(&zmsg, msg.body.size());
    std::memcpy(zmq_msg_data(&zmsg), msg.body.data(), msg.body.size());
    int rc = zmq_msg_send(&zmsg, socket_, ZMQ_DONTWAIT);
    zmq_msg_close(&zmsg);
    return rc == static_cast<int>(msg.body.size());
}

std::optional<FleetMessage> ZeroMqTransport::try_recv() {
    zmq_msg_t zmsg;
    if (zmq_msg_init(&zmsg) != 0) return std::nullopt;
    int rc = zmq_msg_recv(&zmsg, socket_, ZMQ_DONTWAIT);
    if (rc < 0) { zmq_msg_close(&zmsg); return std::nullopt; }
    FleetMessage m;
    m.body.assign(static_cast<const uint8_t*>(zmq_msg_data(&zmsg)),
                 static_cast<const uint8_t*>(zmq_msg_data(&zmsg)) + rc);
    zmq_msg_close(&zmsg);
    return m;
}

void* make_context() { return zmq_ctx_new(); }
void  destroy_context(void* ctx) { if (ctx) zmq_ctx_term(static_cast<zmq_context_t*>(ctx)); }

} // namespace phenotype
