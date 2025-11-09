# GoodFoods AI Reservation Agent

An intelligent AI-powered restaurant reservation system that provides natural conversation interface for booking tables, checking availability, and getting personalized restaurant recommendations.

##  Features

- Natural language restaurant booking interface
- Intelligent restaurant recommendations based on preferences
- Real-time availability checking
- Automated email notifications
- Customer preference tracking
- Analytics and reporting
- Multi-location support

## 🛠️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/goodfoods-ai-reservation-agent.git
   cd goodfoods-ai-reservation-agent
   ```

2. **Set up Python virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Database Setup**
   ```bash
   python scripts/reset_db.py
   python scripts/load_restaurants.py
   python scripts/add_opening_hours.py
   ```

5. **Start the application**
   ```bash
   python app/main.py
   ```

## 🧠 Prompt Engineering Approach

Our AI agent uses a two-stage architecture for natural and accurate responses:

### 1. Planner Agent
- Determines the next best action based on conversation context
- Uses strict JSON-only output for reliability
- Maintains conversation state and tracks required booking information
- Makes decisions based on available tools and user input completeness

### 2. Responder Agent
- Converts structured data into natural, engaging responses
- Maintains consistent personality and tone
- Uses context-aware response templates
- Guides users through the booking process naturally

### Key Principles:
- **Clear Task Boundaries**: Each agent has a specific role (planning vs. responding)
- **Structured Output**: Strict JSON format for internal communication
- **Natural Conversation Flow**: Human-like responses without exposing system mechanics
- **Progressive Information Gathering**: Intelligent handling of missing information

## 💬 Example Conversations

### Journey 1: Quick Booking
```
User: Hi, I'd like to book a table for 4 at GoodFoods Bistro tomorrow
Agent: I'd be happy to help you book a table at GoodFoods Bistro. What time would you prefer for your party of 4 tomorrow?

User: Around 7 PM would be great
Agent: Let me check availability... Great news! I found tables available at 6:45 PM and 7:15 PM tomorrow. Which time works better for you?

User: 7:15 works perfect
Agent: Excellent choice! I just need your email address to confirm the reservation.

User: john.doe@email.com
Agent: Perfect! I've confirmed your reservation for 4 people at GoodFoods Bistro tomorrow at 7:15 PM. A confirmation email has been sent to john.doe@email.com. Looking forward to having you!
```

### Journey 2: Restaurant Discovery
```
User: I'm looking for a nice Italian place in Koramangala
Agent: Here are some Italian spots you might love:
- GoodFoods Bistro: Authentic cuisine, cozy atmosphere (₹₹)
- GoodFoods Trattoria: Family-style dining, great pizzas (₹₹₹)
Would you like to know more about either of these?

User: Tell me more about GoodFoods Bistro
Agent: GoodFoods Bistro is known for their handmade pasta and wood-fired pizzas. They have a lovely outdoor seating area and are perfect for both casual dinners and special occasions. Would you like to check table availability?
```

## 💼 Business Strategy

### Target Market
- Urban professionals seeking convenient dining reservations
- Restaurant owners wanting to optimize table management
- Food enthusiasts looking for personalized recommendations

### Value Proposition
1. **For Customers**
   - Natural conversation interface
   - Personalized restaurant recommendations
   - Seamless booking experience
   - Instant confirmation

2. **For Restaurants**
   - Optimized table utilization
   - Reduced no-shows
   - Customer preference insights
   - Automated management

### Growth Strategy
1. **Phase 1: Core Experience**
   - Perfect the booking experience
   - Build restaurant network
   - Gather user feedback

2. **Phase 2: Enhancement**
   - Add personalization features
   - Implement loyalty program
   - Expand restaurant analytics

3. **Phase 3: Expansion**
   - Geographic expansion
   - Integration with other platforms
   - Advanced AI features

## Performance Metrics
- Conversation completion rate
- Booking conversion rate
- User satisfaction scores
- Restaurant partner satisfaction
- System response time
- Booking accuracy
