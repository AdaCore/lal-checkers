procedure Test is
   type Integer_Access is access Integer;

   type Point is record
      x : Integer;
      y : Integer;
   end record;

   b : Boolean;
   x : Integer_Access;
   p : Point := (1, 2);

   function Deref(ptr: Integer_Access) return Integer is
      lel : Point := (24, 52);
      y : Integer_Access := lel.x'Access;
   begin
      return ptr.all;
   end Test;

begin
   x := p.x'Access;
   p.y := Deref(x);
end Ex1;
