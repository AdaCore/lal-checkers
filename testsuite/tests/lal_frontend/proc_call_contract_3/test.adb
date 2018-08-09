procedure Test is
   type Point is record
      x : Integer;
      y : Integer;
      z : Integer;
   end record;

   function F(x: Integer) return Integer
      with Post => F'Result = x + 1;

   p : Point := (2, 3, 12);
begin
   p.y := F(p.y);
end Ex1;
