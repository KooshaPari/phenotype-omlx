#pragma once
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace phenotype {

enum class Role { PUSH, PULL };

struct FleetMessage {
    std::vector<uint8_t> body;
    uint64_t            src_id = 0;
    uint64_t            dst_id = 0;
};

class ZeroMqTransport {
public:
    ZeroMqTransport(void* context, std::string_view endpoint, Role role);
    ~ZeroMqTransport();
    bool                          send(const FleetMessage& msg);
    std::optional<FleetMessage>   try_recv();
private:
    void*         context_ = nullptr;
    std::string   endpoint_;
    Role          role_;
    void*         socket_  = nullptr; // zmq_socket_t
};

void* make_context();
void  destroy_context(void* ctx);

} // namespace phenotype

extern "C" {
    void* phenotype_zeromq_make(void* ctx, const char* endpoint, int is_pull);
    int   phenotype_zeromq_send(void* handle, const uint8_t* data, size_t len);
    int   phenotype_zeromq_try_recv(void* handle, uint8_t** out_data, size_t* out_len, uint64_t* src_id);
    void  phenotype_zeromq_destroy(void* handle);
    void* phenotype_zeromq_default_context();
    void  phenotype_zeromq_shutdown(void* ctx);
}
