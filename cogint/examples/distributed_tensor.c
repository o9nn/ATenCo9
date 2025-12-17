/**
 * Distributed Tensor Example
 * 
 * Demonstrates distributed tensor operations using 9P protocol.
 */

#include <cogint/cogint.h>
#include <cogint/cog9p.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    printf("=== CogInt Distributed Tensor Example ===\n\n");
    printf("Note: This example requires a 9P server setup.\n");
    printf("For now, showing the basic API usage pattern:\n\n");
    
    CogContext *ctx = cogint_init();
    if (!ctx) {
        fprintf(stderr, "Failed to initialize CogInt context\n");
        return 1;
    }
    
    printf("1. Initialize 9P client connection\n");
    printf("   // cog9p_connect(\"tcp!server:564\")\n\n");
    
    printf("2. Create local tensor\n");
    int64_t shape[] = {100, 100};
    CogTensor *local_tensor = cog_tensor_create(ctx, shape, 2, COG_DTYPE_FLOAT32);
    printf("   ✓ Local tensor created (100x100)\n\n");
    
    printf("3. Export tensor via 9P\n");
    printf("   // cog9p_export_tensor(local_tensor, \"/tensors/my_tensor\")\n\n");
    
    printf("4. Remote nodes can now access the tensor\n");
    printf("   // Remote: cog9p_mount_tensor(\"/tensors/my_tensor\")\n\n");
    
    printf("5. Perform distributed operations\n");
    printf("   // cog9p_distributed_matmul(tensor_a, tensor_b)\n\n");
    
    cog_tensor_free(local_tensor);
    cogint_shutdown(ctx);
    
    printf("=== Example completed ===\n");
    return 0;
}
