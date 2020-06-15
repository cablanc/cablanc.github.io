<ul>
  {% for post in site.posts %}
  {% if post.categories contains 'courses' %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
      {{ post.excerpt }}
    </li>
  {% endif %}
  {% endfor %}
</ul>