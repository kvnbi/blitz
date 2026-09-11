BIN      ?= blitz
CXX      ?= clang++
SRCDIR    = src
SOURCES   = $(SRCDIR)/main.cpp $(SRCDIR)/bitboard.cpp $(SRCDIR)/position.cpp \
            $(SRCDIR)/movegen.cpp $(SRCDIR)/movepick.cpp $(SRCDIR)/evaluate.cpp \
            $(SRCDIR)/search.cpp $(SRCDIR)/thread.cpp $(SRCDIR)/tt.cpp \
            $(SRCDIR)/timeman.cpp $(SRCDIR)/uci.cpp $(SRCDIR)/misc.cpp \
            $(SRCDIR)/bench.cpp $(SRCDIR)/datagen.cpp $(SRCDIR)/nnuecheck.cpp \
            $(SRCDIR)/nnue/network.cpp
OBJECTS   = $(SOURCES:.cpp=.o)
DEPS      = $(OBJECTS:.o=.d)

CXXFLAGS ?= -std=c++20 -O3 -DNDEBUG -flto -fno-exceptions -fno-rtti \
            -Wall -Wextra -Wno-unused-parameter -MMD -MP -I$(SRCDIR)
LDFLAGS  ?= -flto -pthread

HL ?= 1024
CXXFLAGS += -DBLITZ_HL=$(HL)

ifneq ($(shell cat .hlstamp 2>/dev/null),$(HL))
    $(shell rm -f $(OBJECTS) $(DEPS) .hlstamp)
endif

UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),arm64)
    CXXFLAGS += -mcpu=native
else
    CXXFLAGS += -march=native
endif

all: $(BIN)

$(OBJECTS): .hlstamp

.hlstamp:
	@echo "$(HL)" > $@

$(BIN): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)

debug: CXXFLAGS := $(filter-out -O3 -DNDEBUG -flto,$(CXXFLAGS)) -O1 -g -fsanitize=address,undefined
debug: LDFLAGS := -pthread -fsanitize=address,undefined
debug: clean $(BIN)

perft: $(SRCDIR)/perft.cpp $(SRCDIR)/position.cpp $(SRCDIR)/movegen.cpp $(SRCDIR)/bitboard.cpp
	$(CXX) $(CXXFLAGS) $^ -o perft $(LDFLAGS)

bench: $(BIN)
	./$(BIN) bench

test: $(BIN) perft
	./tools/run_tests.sh

clean:
	rm -f $(OBJECTS) $(DEPS) $(BIN) perft .hlstamp

-include $(DEPS)
.PHONY: all clean debug bench test
