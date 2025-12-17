#include <cogint/cogint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int main(void) {
    printf("Running CogInt Basic Tests...\n");
    
    // Test 1: Initialize CogInt context
    printf("Test 1: Context initialization...");
    CogContext *ctx = cogint_init();
    assert(ctx != NULL);
    printf(" PASSED\n");
    
    // Test 2: Create a simple tensor
    printf("Test 2: Tensor creation...");
    int64_t shape[] = {2, 3};
    CogTensor *t = cog_tensor_create(ctx, shape, 2, COG_DTYPE_FLOAT32);
    assert(t != NULL);
    printf(" PASSED\n");
    
    // Test 3: Get tensor properties
    printf("Test 3: Tensor properties...");
    assert(cog_tensor_ndim(t) == 2);
    assert(cog_tensor_size(t, 0) == 2);
    assert(cog_tensor_size(t, 1) == 3);
    assert(cog_tensor_dtype(t) == COG_DTYPE_FLOAT32);
    printf(" PASSED\n");
    
    // Test 4: Cleanup
    printf("Test 4: Resource cleanup...");
    cog_tensor_free(t);
    cogint_shutdown(ctx);
    printf(" PASSED\n");
    
    printf("\nAll basic tests passed!\n");
    return 0;
}
