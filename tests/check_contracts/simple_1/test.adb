procedure Ex1 is
   type Point is record
      x : Integer;
      y : Integer;
      z : Integer;
   end record;

   function F(x : in out Integer) return Integer
      with Pre => x < 50;

   p : Point := (2, 3, 12);
begin
   p.y := F(p.x);
end Ex1;
