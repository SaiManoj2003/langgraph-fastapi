# LangGraph FastAPI Application

This is a **Dockerized FastAPI** application that uses **LangGraph** for building AI workflows. 
The application first **classifies the user’s intent** and routes the request to the appropriate **retrieval node**,  
where relevant information is fetched from the knowledge base. The retrieved content is then passed to a  
**generation node**, which formulates and returns the final answer to the user.
Follow the steps below to build and run the app using Docker.

---

## 📦 Prerequisites
- [Docker](https://docs.docker.com/get-docker/) installed
- [Docker Compose](https://docs.docker.com/compose/install/) installed

---

## 🚀 Running the Application

1. **Clone the repository**  
   ```bash
   git clone https://github.com/SaiManoj2003/langgraph-fastapi.git
   cd langgraph-fastapi
   
2. **Build and run with Docker Compose**
   ```bash
   docker compose up --build

3. **Access the application**
   Once the container is running, open your browser and go to:
   http://0.0.0.0:8000

## API Usage

### Endpoint: `/chat`
Handles user questions and streams the generated response from the LangGraph application.

**Method:** `POST`  
**URL:** `http://0.0.0.0:8000/chat`  

---

### Request Body (JSON)
| Field         | Type   | Description                                     | Required |
|---------------|--------|-------------------------------------------------|----------|
| `user_question` | string | The question you want to ask the system         | Yes      |
| `thread_id`     | string | (Optional) Thread ID to maintain conversation context | No       |

**Example:**
```json
{
  "user_question": "How do you add AI Services in semantic kernel",
  "thread_id": "123e4567-e89b-12d3-a456-426614174000"
}
